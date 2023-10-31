import git
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import json


def flatten(params):
    arrays = jtu.tree_leaves(params)
    assert all(a.dtype == arrays[0].dtype for a in arrays[1:])
    return jnp.concatenate(
        [param.reshape(-1) for param in jtu.tree_leaves(params)],
        dtype=arrays[0].dtype,
    )


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
    learning_rate=0.01,
    batch_size=56,
    b1_min=0.85,
    b1_max=0.95,
    b2=0.99,
    epsilon=1e-5,
    sam_stride=0.01,
    image_size=512,
    train_steps=1024,
    warmup_steps=128,
    weight_decay=0.00,
    save_every=1024,
):
    config = locals().copy()
    import time
    import itertools as it
    import rich.progress as rp
    from istrm import load_chunks, fetch_posts_meta
    from functools import partial
    from diffusers import FlaxAutoencoderKL
    from flax.training import train_state
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

    class TrainState(train_state.TrainState):
        moment: jax.Array
        scales: jax.Array
        err_st: optax.EmaState
        losses: optax.Params
        rng: jax.random.PRNGKey

    with jax.default_device(jax.devices("cpu")[0]):
        vae, params = FlaxAutoencoderKL.from_pretrained(
            model_slug, dtype=jnp.bfloat16, subfolder="vae"
        )
        params = jtu.tree_map(lambda a: a.astype(jnp.bfloat16), params)
        params = {"params": params, "alpha": jnp.zeros([], dtype=jnp.bfloat16)}
        errors = ("loss", "reconstruction_err", "blurriness_err")
        losses = dict(zip(errors, jnp.zeros([len(errors)], dtype=jnp.bfloat16)))
        tstate = TrainState.create(
            apply_fn=vae.__call__,
            params=params,
            tx=optax.scale(1.0),  # no-op
            moment=jnp.zeros_like(flatten(params), dtype=jnp.bfloat16),
            scales=jnp.ones_like(flatten(params), dtype=jnp.bfloat16),
            err_st=optax.ema(0.9).init(losses),
            losses=losses,
            rng=jax.random.PRNGKey(42),
        )

    shards = jsh.PositionalSharding(jax.devices())
    # lsched = optax.linear_onecycle_schedule(train_steps, learning_rate)
    lsched = optax.warmup_cosine_decay_schedule(
        1e-5,
        learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=train_steps - warmup_steps,
    )

    def msched(step):
        return b1_min + (b1_max - b1_min) * (lsched(step) / learning_rate)

    @jax.checkpoint
    def objective(params, rng, images):
        def laplacian(gamma):
            signal = jnp.pad(gamma, ((1, 1), (1, 1)), mode="reflect")
            kernel = jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=gamma.dtype)
            return jax.scipy.signal.convolve2d(signal, kernel, mode="valid")

        alpha = params.pop("alpha")
        rngs = dict(zip(["params", "dropout", "gaussian"], jax.random.split(rng, 3)))
        output = vae.apply(
            {"params": params["params"]},
            images,
            rngs=rngs,
            deterministic=True,
        ).sample
        reconstruction_err = optax.l2_loss(images, output).mean()
        edge_maps = map(
            lambda images: jax.vmap(laplacian)(
                rearrange(images.astype(jnp.bfloat16), "... d h w -> (... d) h w")
            ),
            (images, output),
        )
        blurriness_err = optax.l2_loss(*edge_maps).mean()
        total_err = (
            jax.nn.sigmoid(alpha) * reconstruction_err
            + (1 - jax.nn.sigmoid(alpha)) * blurriness_err
        )
        params["alpha"] = alpha
        return total_err, (reconstruction_err, blurriness_err)

    @partial(jax.jit, donate_argnums=0, out_shardings=shards.replicate())
    def train_step(tstate, images):
        images = rearrange(
            (images / 255.0).astype(jnp.bfloat16), "... h w d -> ... d h w"
        )
        rng, dropout_key, noising_key = jax.random.split(tstate.rng, 3)
        params = flatten(tstate.params)
        scales = tstate.scales
        ascent = jax.random.normal(noising_key, params.shape, dtype=params.dtype)
        ascent /= jnp.sqrt(scales * np.prod(images.shape[:-3])) + epsilon
        grad, _ = jax.grad(objective, has_aux=True)(
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
        scales = b2 * scales + (1 - b2) * (
            jnp.sqrt(scales) * jnp.abs(grad) + 0.1 + weight_decay
        )
        (loss, (reconstruction_err, blurriness_err)), grad = jax.value_and_grad(
            objective, has_aux=True
        )(unflatten(params + ascent, tstate.params), dropout_key, images)
        grad = flatten(grad)
        step_inc = optax.safe_int32_increment(tstate.step)
        b1 = msched(step_inc).astype(params.dtype)
        moment = optax.update_moment(grad, tstate.moment, decay=b1, order=1)
        # TODO: figure out a way to perform aesthetic conditioning. GroupNorm FiLM?
        # bSAM: algorithm 1
        losses = {
            "loss": loss,
            "reconstruction_err": reconstruction_err,
            "blurriness_err": blurriness_err,
        }
        losses, err_st = optax.ema(0.9).update(losses, tstate.err_st)
        params = flatten(tstate.params)
        params -= (
            lsched(step_inc).astype(params.dtype)
            * optax.bias_correction(moment, b1, step_inc)
            / (optax.bias_correction(scales, b2, step_inc) + epsilon)
        )
        params = unflatten(params, tstate.params)
        params["alpha"] = alpha
        tstate = replace(
            tstate,
            params=params,
            moment=moment,
            scales=scales,
            err_st=err_st,
            losses=losses,
            rng=rng,
            step=tstate.step + 1,
        )
        return tstate

    posts = pl.read_parquet(posts_path)
    loader = load_chunks(posts, batch_size=batch_size)
    loader = iter(loader)
    loader = it.islice(loader, train_steps)
    tstate = jax.device_put(tstate, shards.replicate())

    repo = git.Repo(".")
    head = repo.git.rev_parse(repo.head.commit.hexsha, short=8)
    with rp.Progress(
        "loss: {task.fields[loss]:.3g}",
        *rp.Progress.get_default_columns()[:-2],
        rp.MofNCompleteColumn(),
        rp.TimeElapsedColumn(),
    ) as progress, open("fluffyvae.jsonl", "a+", encoding="utf8") as log:
        task = progress.add_task(
            "training...",
            start=False,
            total=train_steps,
            loss=float("nan"),
        )
        for images, meta in loader:
            images = jax.device_put(images, shards.reshape(-1, 1, 1, 1))
            tstate = train_step(tstate, images)
            losses = {k: float(v) for k, v in jax.device_get(tstate.losses).items()}
            step = jax.device_get(tstate.step).astype(int)
            if step == train_steps or step % save_every == 0:
                vae.save_pretrained(
                    "fluffyvae.ckpt",
                    params=jax.device_get(tstate.params["params"]),
                    to_pytorch=True,
                    safe_serialization=True,
                )
            progress.start_task(task)
            progress.update(task, advance=1, loss=losses["loss"])
            record = {
                "head": head,
                "step": int(step),
                "time": time.time(),
                **losses,
            }
            record = json.dumps(record)
            log.write(f"{record}\n")
            log.flush()


if __name__ == "__main__":
    import polars as pl

    def plot_latest_run():
        import seaborn as sns

        data = (
            pl.scan_ndjson("fluffyvae.jsonl")
            .with_columns(
                (pl.col("step").diff().fill_null(1) != 1).cumsum().alias("run")
            )
            .filter(pl.col("run") == pl.col("run").max())
            .melt(
                id_vars=["step"],
                value_vars=["loss", "reconstruction_err", "blurriness_err"],
            )
            .collect()
        )
        plot = sns.lineplot(data=data.to_pandas(), x="step", y="value", hue="variable")
        return plot

    plot = plot_latest_run()
    main()
    # fire.Fire(main)
