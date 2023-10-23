import fire
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


def flatten(params):
    return jnp.concatenate([param.reshape(-1) for param in jtu.tree_leaves(params)])


def unflatten(flat, updates):
    updates_flat, treedef = jtu.tree_flatten(updates)
    offsets = []
    for update in updates_flat:
        if offsets:
            offsets.append(update.size + offsets[-1])
        else:
            offsets.append(update.size)
    del offsets[-1]
    flat_split = jnp.split(flat, offsets)
    reshaped = [
        jnp.reshape(flat_update, update.shape)
        for flat_update, update in zip(flat_split, updates_flat)
    ]
    return jtu.tree_unflatten(treedef, reshaped)


def main(
    model_slug="stabilityai/stable-diffusion-xl-base-1.0",
    # cache_path="./base-vae",
    posts_path="./posts.parquet",
    learning_rate=2e-6,
    batch_size=8,
    b1_min=0.85,
    b1_max=0.95,
    b2=0.99,
    epsilon=1e-5,
    sam_stride=0.05,
    image_size=512,
    train_steps=1024,
    warmup_steps=128,
    weight_decay=1e-2,
    save_every=1024,
):
    import itertools as it
    import rich.progress as rp
    import orbax.checkpoint
    from istrm import load_chunks, fetch_posts_meta
    from functools import partial
    from diffusers import FlaxAutoencoderKL
    from flax.training import train_state, orbax_utils
    from dataclasses import replace
    import jax
    from einops import rearrange
    import jax.sharding as jsh
    import optax
    import polars as pl
    import os.path as osp

    if not osp.exists(posts_path):
        posts = fetch_posts_meta()
        posts.write_parquet(posts_path)
    else:
        posts = pl.read_parquet(posts_path)

    with jax.default_device(jax.devices("cpu")[0]):
        vae, params = FlaxAutoencoderKL.from_pretrained(
            model_slug, dtype=jnp.bfloat16, subfolder="vae"
        )
        params = {"params": params, "alpha": jnp.zeros([])}

    ckpointer = orbax.checkpoint.PyTreeCheckpointer()
    shards = jsh.PositionalSharding(jax.devices())
    params = jax.device_put(params, shards.replicate())

    lsched = optax.warmup_cosine_decay_schedule(
        1e-8,
        learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=train_steps - warmup_steps,
    )

    def msched(step):
        return b1_min + (b1_max - b1_min) * (lsched(step) / learning_rate)

    @jax.checkpoint
    def objective(params, rng, images):
        alpha = params.pop("alpha")
        rngs = dict(zip(["params", "dropout", "gaussian"], jax.random.split(rng, 3)))
        output = vae.apply(
            {"params": params["params"]},
            images,
            rngs=rngs,
            deterministic=False,
        ).sample
        reproduction_err = optax.l2_loss(images, output)
        spectral_maps = map(
            lambda images: jax.vmap(jnp.fft.rfft2)(
                rearrange(images.astype(jnp.float32), "... d h w -> (... d) h w")
            ).real.astype(jnp.bfloat16),
            (images, output),
        )
        spectral_maps = jnp.stack(list(spectral_maps), axis=0)
        spectral_maps = spectral_maps[
            ..., : spectral_maps.shape[-2] // 2, : spectral_maps.shape[-1] // 2
        ]
        spectral_maps = jnp.log(jnp.abs(spectral_maps))
        blurriness_err = optax.l2_loss(*spectral_maps)
        total_err = (
            jax.nn.sigmoid(alpha) * reproduction_err.mean()
            + (1 - jax.nn.sigmoid(alpha)) * blurriness_err.mean()
        )
        params["alpha"] = alpha
        return total_err

    class TrainState(train_state.TrainState):
        moment: jax.Array
        scales: jax.Array
        loss: jax.Array
        rng: jax.random.PRNGKey

    # Initialize our training
    tstate = TrainState.create(
        apply_fn=vae.__call__,
        params=params,
        tx=optax.scale(1.0),  # no-op
        moment=jnp.zeros_like(flatten(params), dtype=jnp.bfloat16),
        scales=jnp.ones_like(flatten(params), dtype=jnp.bfloat16),
        loss=0.0,
        rng=jax.random.PRNGKey(42),
    )

    @partial(jax.jit, donate_argnums=0, out_shardings=shards.replicate())
    def train_step(tstate, images):
        images = rearrange(
            (images / 255.0).astype(jnp.bfloat16), "... h w d -> ... d h w"
        )
        rng, dropout_key, noising_key = jax.random.split(tstate.rng, 3)
        params = flatten(tstate.params)
        scales = tstate.scales
        ascent = jax.random.normal(noising_key, params.shape, dtype=params.dtype)
        ascent /= scales * np.prod(images.shape[:-3]) + epsilon
        loss, grad = jax.value_and_grad(objective)(
            unflatten(params + ascent, tstate.params), dropout_key, images
        )
        alpha = (
            tstate.params["alpha"]
            + 0.01 * grad["alpha"]
            - 0.01 * tstate.params["alpha"]
        )
        grad = flatten(grad)
        grad /= optax.safe_norm(grad, epsilon) * 1.0
        grad += weight_decay * params
        ascent = sam_stride * grad / scales
        scales = b2 * scales + (1 - b2) * jnp.sqrt(scales) * jnp.abs(grad)
        grad = jax.grad(objective)(
            unflatten(params + ascent, tstate.params), dropout_key, images
        )
        grad = flatten(grad)
        moment = msched(tstate.step) * tstate.moment + (1 - msched(tstate.step)) * grad
        # TODO: figure out a way to perform aesthetic conditioning. GroupNorm FiLM?
        # bSAM: algorithm 1
        params = flatten(tstate.params) - (
            lsched(tstate.step)
            * moment
            / (scales + epsilon)
            # * optax.bias_correction(moment, msched(tstate.step), tstate.step)
            # / (optax.bias_correction(scales, b2, tstate.step) + epsilon)
        )
        params = unflatten(params, tstate.params)
        params["alpha"] = alpha
        avg_step = jnp.minimum(tstate.step, save_every)
        loss_avg = (tstate.loss * avg_step + loss) / (avg_step + 1)
        tstate = replace(
            tstate,
            params=params,
            moment=moment,
            scales=scales,
            loss=loss_avg,
            rng=rng,
            step=tstate.step + 1,
        )
        return tstate

    posts = pl.read_parquet(posts_path)
    loader = load_chunks(
        posts,
        batch_size=batch_size,
        height=image_size,
        width=image_size,
    )
    loader = it.islice(loader, train_steps)
    loader = it.starmap(
        lambda images, meta: (
            jax.device_put(images, shards.reshape(-1, 1, 1, 1)),
            meta,
        ),
        loader,
    )
    with rp.Progress(
        "loss: {task.fields[loss]:.3g}",
        *rp.Progress.get_default_columns()[:-2],
        rp.MofNCompleteColumn(),
        rp.TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "training...",
            start=False,
            total=train_steps,
            loss=float("nan"),
        )
        for images, meta in loader:
            tstate = train_step(tstate, images)
            loss = jax.device_get(tstate.loss).astype(float)
            progress.start_task(task)
            progress.update(task, advance=1, loss=loss)
            step = jax.device_get(tstate.step).astype(int)
            if step == train_steps or step % save_every == 0:
                ckpt_target = tstate.params
                ckpt_config = orbax_utils.save_args_from_target(ckpt_target)
                ckpointer.save(
                    "params.ckpt", ckpt_target, save_args=ckpt_config, force=True
                )
                del ckpt_target


if __name__ == "__main__":
    main()
    # fire.Fire(main)
