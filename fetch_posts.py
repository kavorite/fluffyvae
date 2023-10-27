from istrm import client, fetch_blob_urls
import rich.progress as rp
import multiprocessing.pool as mpp
import polars as pl


def _download_posts_shard(url):
    blob = client.get(url).content
    head_idx = blob.index(b"\n")
    return blob[:head_idx], blob[head_idx:]


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
    posts.write_parquet("posts.parquet")
