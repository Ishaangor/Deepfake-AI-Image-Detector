"""Fetch a stratified, seeded subset of CommunityForensics-Eval (CompEval) as ORIGINAL image files. Evaluation only.

    python cf_eval_fetch.py            # -> data/commfor/eval/*.{png,jpg,tif,...}, artifacts/cf_eval_sel.parquet

CompEval = 21 generators (none appear in the Small training set) + reals from LAION, RAISE, COCO, FFHQ, ImageNet.
License CC BY-NC-SA 4.0, non-commercial research. Row groups are almost pure per generator, so whole row groups
(<=25 images) are range-read without downloading the remaining ~200 GB.
Quotas (row groups, chosen by seed before looking at any image): generators 3 each (~75 images), reals LAION 20,
COCO 6, FFHQ 6, ImageNet 6, RAISE 4 (RAISE TIFFs are ~25 MB each).
"""
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "OwensLab/CommunityForensics-Eval"
OUT = os.path.join("data", "commfor", "eval")
SEL = os.path.join("artifacts", "cf_eval_sel.parquet")
SEED = 20261009
EXT = {"PNG": "png", "JPEG": "jpg", "MPO": "jpg", "TIFF": "tif", "WEBP": "webp", "GIF": "gif"}
REAL_QUOTA = {"LAION": 20, "coco": 6, "ffhq": 6, "imagenet": 6, "RAISE": 4}
GEN_QUOTA = 3


def choose():
    e = pd.read_parquet(os.path.join("artifacts", "cf_eval_meta.parquet"))
    e["key"] = e["model_name"].where(e.label == 1, "REAL:" + e.real_source)
    rng = random.Random(SEED)
    chosen = []
    for key, d in e.groupby("key"):
        groups = sorted(d.groupby(["shard", "row_group"]).size().index.tolist())
        k = REAL_QUOTA.get(key.split(":")[-1], GEN_QUOTA) if key.startswith("REAL:") else GEN_QUOTA
        chosen += [(key, s, g) for s, g in rng.sample(groups, min(k, len(groups)))]
    return e, chosen


def fetch(job, fs):
    key, shard, rg = job
    slug = re.sub(r"[^A-Za-z0-9]+", "_", key)
    marker = os.path.join(OUT, f".done_{slug}_{shard[-22:-8]}_{rg}")
    if os.path.exists(marker):
        return
    for attempt in range(5):
        try:
            with fs.open(f"datasets/{REPO}/{shard}", "rb") as f:
                t = pq.ParquetFile(f).read_row_group(rg, columns=["image_name", "format", "image_data"]).to_pylist()
            for i, r in enumerate(t):
                fn = f"{slug}__{shard[-22:-8]}_{rg}_{i}.{EXT.get(r['format'], 'bin')}"
                with open(os.path.join(OUT, fn), "wb") as o:
                    o.write(r["image_data"])
            open(marker, "w").close()
            print("ok", key, shard[-22:-8], rg, len(t), flush=True)
            return
        except Exception as ex:
            print("retry", key, rg, attempt, repr(ex)[:100], flush=True)
    print("FAILED", job, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    e, chosen = choose()
    sel = pd.concat([e[(e.key == k) & (e.shard == s) & (e.row_group == g)] for k, s, g in chosen])
    sel["slug"] = sel.key.map(lambda k: re.sub(r"[^A-Za-z0-9]+", "_", k))
    sel["file"] = [f"{sl}__{sh[-22:-8]}_{rg}_{i}.{EXT.get(fm, 'bin')}" for sl, sh, rg, i, fm in
                   zip(sel.slug, sel.shard, sel.row_group, sel.row_in_group, sel.format)]
    sel.to_parquet(SEL)
    print(len(chosen), "row groups,", len(sel), "images;", sel.groupby("key").size().to_dict(), flush=True)
    fs = HfFileSystem()
    # RAISE (very large) last and sequential, everything else 2 threads
    big = [j for j in chosen if j[0] == "REAL:RAISE"]
    rest = [j for j in chosen if j[0] != "REAL:RAISE"]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: fetch(j, fs), rest))
    for j in big:
        fetch(j, fs)
    print("EVAL FETCH COMPLETE", flush=True)
