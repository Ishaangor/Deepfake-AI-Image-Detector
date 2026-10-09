"""Second-generation data plan after the first CompEval result (see docs/EXPERIMENTS.md).

The Small-only detector scored ROC-AUC 0.69 on CompEval-v1 (21 unseen generators, web-photo reals). That sample is now
RETIRED as a test set for any model trained with CompEval data; it stays as the documented zero-shot result of the
Small-only models. This script builds a new, disjoint split from the *remaining* CompEval row groups:

  train / val   : families chosen by hash  (md5("family:<name>") -> u)       u<0.35 bench, <0.5 val, else train
  bench         : HELD-OUT FAMILIES (Midjourney, FLUX, Imagen, stable_cascade) + held-out real row groups: the new
                  independent benchmark "CompEval-v2". Never used for training, selection or calibration.
  reals         : whole row groups are assigned to train/val/calib/bench, so no image of a row group is on both sides.
Row groups already used by CompEval-v1 are excluded from everything. Quotas are fixed below, before any of these images
is looked at. Originals are saved to data/commfor/ce/ (git-ignored); selection to artifacts/ce_sel.parquet.

    python ce_fetch.py
"""
import hashlib
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "OwensLab/CommunityForensics-Eval"
OUT = os.path.join("data", "commfor", "ce")
SEL = os.path.join("artifacts", "ce_sel.parquet")
SEED = 20261010
EXT = {"PNG": "png", "JPEG": "jpg", "MPO": "jpg", "TIFF": "tif", "WEBP": "webp", "GIF": "gif"}
FAMILIES = [("LCM_lora", "LCM"), ("idjourney", "Midjourney"), ("Firefly", "Firefly"), ("Ideogram", "Ideogram"),
            ("FLUX", "FLUX"), ("Dalle", "Dalle"), ("Imagen", "Imagen")]
FAKE_GROUPS = dict(train=16, val=6, bench=14)                       # row groups (<=25 images) per generator
REAL_GROUPS = {"LAION": dict(train=200, val=12, calib=12, bench=40), "imagenet": dict(train=12, val=4, calib=4, bench=10),
               "coco": dict(bench=6), "ffhq": dict(bench=6), "RAISE": dict(bench=6)}


def family(m):
    return next((v for k, v in FAMILIES if k in m), m)


def fam_role(f):
    u = int(hashlib.md5(("family:" + f).encode()).hexdigest()[:8], 16) / 2 ** 32
    return "bench" if u < 0.35 else ("val" if u < 0.5 else "train")


def choose():
    e = pd.read_parquet(os.path.join("artifacts", "cf_eval_meta.parquet"))
    used = set(map(tuple, pd.read_parquet(os.path.join("artifacts", "cf_eval_sel.parquet"))[["shard", "row_group"]].values))
    e["key"] = e.model_name.where(e.label == 1, "REAL:" + e.real_source)
    rng, jobs = random.Random(SEED), []
    for key, d in e.groupby("key"):
        groups = [g for g in sorted(d.groupby(["shard", "row_group"]).size().index.tolist()) if g not in used]
        rng.shuffle(groups)
        if key.startswith("REAL:"):
            quota = REAL_GROUPS.get(key.split(":", 1)[1], {})
        else:
            role = fam_role(family(key))
            quota = {role: FAKE_GROUPS[role]}
        i = 0
        for role, k in quota.items():
            jobs += [(key, s, g, role) for s, g in groups[i:i + k]]
            i += k
    return e, jobs


def fetch(job, fs):
    key, shard, rg, role = job
    slug = re.sub(r"[^A-Za-z0-9]+", "_", key)
    marker = os.path.join(OUT, f".done_{slug}_{shard[-22:-8]}_{rg}")
    if os.path.exists(marker):
        return
    for attempt in range(5):
        try:
            with fs.open(f"datasets/{REPO}/{shard}", "rb") as f:
                t = pq.ParquetFile(f).read_row_group(rg, columns=["format", "image_data"]).to_pylist()
            for i, r in enumerate(t):
                fn = f"{slug}__{shard[-22:-8]}_{rg}_{i}.{EXT.get(r['format'], 'bin')}"
                with open(os.path.join(OUT, fn), "wb") as o:
                    o.write(r["image_data"])
            open(marker, "w").close()
            print("ok", key, role, shard[-22:-8], rg, len(t), flush=True)
            return
        except Exception as ex:
            print("retry", key, rg, attempt, repr(ex)[:100], flush=True)
    print("FAILED", job, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    e, jobs = choose()
    parts = []
    for key, s, g, role in jobs:
        d = e[(e.key == key) & (e.shard == s) & (e.row_group == g)].copy()
        d["role"] = role
        parts.append(d)
    sel = pd.concat(parts)
    sel["slug"] = sel.key.map(lambda k: re.sub(r"[^A-Za-z0-9]+", "_", k))
    sel["file"] = [f"{sl}__{sh[-22:-8]}_{rg}_{i}.{EXT.get(fm, 'bin')}" for sl, sh, rg, i, fm in
                   zip(sel.slug, sel.shard, sel.row_group, sel.row_in_group, sel.format)]
    sel["family"] = [family(m) if lab == 1 else "real" for m, lab in zip(sel.model_name, sel.label)]
    sel.to_parquet(SEL)
    print(len(jobs), "row groups,", len(sel), "images")
    print(pd.crosstab(sel.key, sel.role).to_string(), flush=True)
    fs = HfFileSystem()
    big = [j for j in jobs if j[0] == "REAL:RAISE"]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: fetch(j, fs), [j for j in jobs if j[0] != "REAL:RAISE"]))
    for j in big:
        fetch(j, fs)
    print("CE FETCH COMPLETE", flush=True)
