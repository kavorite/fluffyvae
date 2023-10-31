from functools import partial
import httpx
from dotenv import dotenv_values, load_dotenv
from stream_unzip import stream_unzip
import polars as pl
import fire
import multiprocessing.pool as mpp
import threading
import os
import numpy as np
import itertools as it
import rich.progress as rp
import albumentations as iaa
from typing import Callable, Iterator

load_dotenv()
import cv2

cv2.setNumThreads(os.cpu_count())

_token = dotenv_values()["HF_TOKEN"]
client = httpx.Client(
    base_url="https://huggingface.co/",
    headers={"Authorization": f"Bearer {_token}"},
    follow_redirects=True,
    timeout=httpx.Timeout(timeout=None),
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


class FivePointCrop:
    def __init__(self, dimens=[512, 512]):
        tgt_h, tgt_w = dimens
        dimens = np.array(dimens)
        self.dimens = dimens

        def _length_slicer(offset, target_length):
            def _slice(length):
                i = length + offset if offset < 0 else offset
                j = i + target_length
                return slice(i, j)

            return _slice

        _left = _length_slicer(0, tgt_w)
        _right = _length_slicer(-tgt_w, tgt_w)
        _top = _length_slicer(0, tgt_h)
        _bottom = _length_slicer(-tgt_h, tgt_h)

        def _corner_crop(vertical_slicer, horizontal_slicer):
            def _crop(image):
                img_h, img_w = image.shape[-3:-1]
                return image[vertical_slicer(img_h), horizontal_slicer(img_w), :]

            return _crop

        def _central(image):
            i, j = np.maximum(np.array(image.shape[-3:-1]) // 2 - dimens // 2, 0)
            cropped = image[i : i + tgt_w, j : j + tgt_w, :]
            return cropped

        self.crops = [_central] + [
            _corner_crop(*slicers)
            for slicers in it.product((_top, _bottom), (_left, _right))
        ]

    def _pad(self, image):
        dimens = self.dimens
        imdims = np.array(image.shape[-3:-1])
        deltas = np.maximum(0, dimens - imdims)
        bottom, left = deltas // 2
        top, right = deltas - deltas // 2
        padding = [(top, bottom), (left, right), (0, 0)]
        padded = np.pad(image, padding)
        return padded

    def __call__(self, image):
        pad = self._pad
        yield from (pad(crop(image)) for crop in self.crops)


def load_chunks(
    posts: pl.DataFrame,
    views: Callable[[np.ndarray], Iterator[np.ndarray]] = FivePointCrop([512, 512]),
    batch_size=32,
    decoder_threads=None,
    prefetch_images=None,
):
    decoder_threads = decoder_threads or min(batch_size, os.cpu_count())
    prefetch_images = prefetch_images or decoder_threads * 2
    buffer_bounds = threading.BoundedSemaphore(prefetch_images)

    # TODO: output the applicable mask or pre-padding dims for each image (for loss)

    def _batched(iterable):
        iterator = iter(iterable)
        yield from iter(lambda: tuple(it.islice(iterator, batch_size)), ())

    posts = posts.sort("md5")

    def _im_decode(triplet):
        file_name, file_size, buffer = triplet
        buffer = np.frombuffer(buffer, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return
        else:
            return file_name, image

    def _metadata(file_names):
        file_names = pl.Series("md5", file_names).cast(str)
        file_names = file_names.str.replace("[.].*$", "")
        # TODO: sort tags by ascending frequency, drop-out, tokenize, truncate
        captions = posts.select(pl.all().take(pl.col("md5").search_sorted(file_names)))
        return captions

    def _content_chunks(endpoint):
        with client.stream(
            "GET",
            endpoint,
        ) as rsp:
            yield from rsp.iter_bytes(1 << 16)

    def _unzip_from(endpoint):
        triplets = (
            (name, size, b"".join(chunks))
            for name, size, chunks in stream_unzip(_content_chunks(endpoint))
        )
        for triplet in triplets:
            buffer_bounds.acquire()
            yield triplet

    with mpp.ThreadPool(decoder_threads) as decoder_pool:
        endpoints = (url for url in fetch_blob_urls() if url.endswith(".zip"))
        unzipped = it.chain.from_iterable(map(_unzip_from, it.cycle(endpoints)))

        def _pairs():
            for pair in decoder_pool.imap(_im_decode, unzipped):
                try:
                    if pair is None:
                        continue
                    else:
                        file_name, image = pair
                        for view in views(image):
                            yield (file_name, view)
                finally:
                    buffer_bounds.release()

        def _collate(pairs):
            file_names, images = zip(*pairs)
            images = np.stack(images)
            return images, _metadata(file_names)

        yield from map(_collate, _batched(_pairs()))


def main(posts_path="./posts.parquet"):
    posts = pl.read_parquet(posts_path)

    for images, metadata in rp.track(load_chunks(posts)):
        pass


if __name__ == "__main__":
    fire.Fire(main)
