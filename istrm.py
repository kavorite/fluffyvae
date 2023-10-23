import httpx
from dotenv import dotenv_values, load_dotenv
from stream_unzip import stream_unzip
import polars as pl
import fire
import multiprocessing.pool as mpp
import os
import numpy as np
import itertools as it
import rich.progress as rp
from prefetch_generator import prefetch


load_dotenv()
import cv2

cv2.setNumThreads(os.cpu_count())

_token = dotenv_values()["HF_TOKEN"]
client = httpx.Client(
    base_url="https://huggingface.co/",
    headers={"Authorization": f"Bearer {_token}"},
    follow_redirects=True,
    timeout=httpx.Timeout(10.0, connect=None),
    limits=httpx.Limits(keepalive_expiry=None),
)
dsslug = "planetexpress/e6"


def ds_blob_url(name, revision="main"):
    return f"datasets/{dsslug}/resolve/{revision}/{name}"


def fetch_blob_urls():
    repo = client.get(f"api/datasets/{dsslug}", params={"full": "full"}).json()

    yield from map(ds_blob_url, (f["rfilename"] for f in repo["siblings"]))


def _download_posts_shard(url):
    blob = client.get(url).content
    head_idx = blob.index(b"\n")
    return blob[:head_idx], blob[head_idx:]


def fetch_posts_meta():
    meta_names = [
        name
        for name in fetch_blob_urls()
        if name.endswith(".csv") and not name.endswith("-full-post.csv")
    ]
    with mpp.ThreadPool() as pool:
        blobs = rp.track(
            pool.imap_unordered(_download_posts_shard, meta_names),
            description="fetch metadata...",
            total=len(meta_names),
        )
        heads, tails = zip(*blobs)
        posts = pl.read_csv(heads[0] + b"".join(tails), infer_schema_length=1 << 16)
        return posts


def load_chunks(
    posts: pl.DataFrame,
    height=512,
    width=512,
    batch_size=32,
    request_threads=1,
    decoder_threads=None,
):
    # TODO: random crops? Res binning/resizing? (This is task parallel, so we can branch)
    def _batched(iterable):
        iterator = iter(iterable)
        yield from iter(lambda: tuple(it.islice(iterator, batch_size)), ())

    posts = posts.sort("md5")
    imdim = np.array([height, width])

    def _im_decode(triplet):
        file_name, file_size, buffer = triplet
        buffer = np.frombuffer(buffer, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)[..., ::-1]
        if image is None:
            return
        else:
            shape = np.array(image.shape[-3:-1])
            y_center, x_center = shape // 2
            image = image[
                max(0, y_center - height // 2) : y_center - (height // -2),
                max(0, x_center - width // 2) : x_center - (width // -2),
            ]
            if np.any(shape < np.array([height, width])):
                width_diff, height_diff = map(int, np.maximum(imdim - shape, 0))
                padding = [
                    (width_diff // 2, -(width_diff // -2)),
                    (height_diff // 2, -(height_diff // -2)),
                    (0, 0),
                ]
                image = np.pad(image, padding)
            return file_name, image

    def _metadata(file_names):
        file_names = pl.Series("md5", file_names).cast(str)
        file_names = file_names.str.replace("[.].*$", "")
        # TODO: sort tags by ascending frequency, drop-out, tokenize, truncate
        captions = posts.select(pl.all().take(pl.col("md5").search_sorted(file_names)))
        return captions

    def _unzip_from(endpoint):
        while True:
            with client.stream("GET", endpoint) as rsp:
                istrm = rsp.iter_bytes(1 << 16)
                triplets = (
                    (name, size, b"".join(chunks))
                    for name, size, chunks in stream_unzip(istrm)
                )
                yield from triplets

    decoder_threads = decoder_threads or min(batch_size, os.cpu_count())
    with (
        mpp.ThreadPool(request_threads) as request_pool,
        mpp.ThreadPool(decoder_threads) as decoder_pool,
    ):

        def _round_robin(*iterables, mapper_fn=map):
            cursors = list(it.islice(map(iter, iterables), batch_size))
            while True:
                try:
                    for i, x in enumerate(mapper_fn(next, cursors)):
                        yield x
                except StopIteration:
                    cursors[i] = iter(iterables[i])

        endpoints = (url for url in fetch_blob_urls() if url.endswith(".zip"))
        unzipped = _round_robin(
            *map(_unzip_from, endpoints),
            mapper_fn=request_pool.imap,
        )
        pairs = decoder_pool.imap(_im_decode, unzipped)
        pairs = (pair for pair in pairs if pair is not None)

        def _collate(pairs):
            file_names, images = zip(*pairs)
            images = np.stack(images)
            return images, _metadata(file_names)

        yield from map(_collate, _batched(pairs))


def main(posts_path="./posts.parquet"):
    posts = pl.read_parquet(posts_path)

    for images, metadata in rp.track(load_chunks(posts, batch_size=32)):
        pass


if __name__ == "__main__":
    fire.Fire(main)