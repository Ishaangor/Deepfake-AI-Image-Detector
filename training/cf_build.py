"""Turn downloaded Community Forensics-Small shards into a compact, bias-controlled training cache.

    python -m training.cf_build select                 # write artifacts/cf_selection.parquet (needs cf_eval_meta.parquet)
    python -m training.cf_build build 123 131 146      # build cache for specific downloaded shards
    python -m training.cf_build build all              # every downloaded manifest shard (JPEG reference shards first)

Per image the cache stores two 256x256 uint8 RGB views of the (possibly re-compressed) image:
  native : an 8-pixel-aligned crop at native resolution (keeps sensor / generator pixel statistics)
  glob   : the whole image, shorter side resized to 256 then centre-cropped (keeps global appearance)

Shortcut controls (decided before any model was trained):
  * Format / JPEG:  the dataset has fakes ~98% PNG vs reals ~45% JPEG. PNG images (all PNG fakes and the PNG real
    sources FFHQ / landscapes) are JPEG-recompressed with probability p so that the share of JPEG-compressed images is
    ~equal in both classes (p_png_real = 0.75, p_fake solved from the selection). Quality is drawn from the empirical
    estimated qualities of the natively-JPEG real images, so the quality distribution is the same for both classes.
  * Resolution:     fakes are mostly 512x512 while reals vary; native crops hide image size, the global view is resized.
  * Generators:     capped at 60 images per SD-derived generator and 500 per hand-collected architecture; generators that also appear in CommunityForensics-Eval are
    EXCLUDED so the independent test stays independent. Generators are split train/val/calib 80/10/10 by hash, so
    validation and calibration measure unseen generators.
  * Splits:         reals split by image hash (COCO, landscapes, FFHQ) or by camera device (VISION) to avoid shared
    scenes. Identity leakage in FFHQ cannot be checked (no identity metadata).
"""
import hashlib
import io
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import detector
from training import parquet_stream
from detector.paths import ARTIFACTS, DATA

RAW = os.path.join(DATA, "commfor", "raw")
CACHE = os.path.join(DATA, "commfor", "cache")
SEL = os.path.join(ARTIFACTS, "cf_selection.parquet")
QPOOL = os.path.join(ARTIFACTS, "cf_qpool.json")
VIEW = 256
P_PNG_REAL = 0.75
CAP_PER_GEN = 60      # Systematic (Stable-Diffusion-derived fine-tunes, ~40-57 images each; cap rarely binds)
CAP_MANUAL = 500      # Manual set: 19 architecturally distinct generators (GANs, DiT, guided diffusion, ...)
SEED = 20261009
STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
            29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
            121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def u01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 2 ** 32


def split_of(u):
    return "train" if u < 0.8 else ("val" if u < 0.9 else "calib")


def est_jpeg_q(im):
    qt = getattr(im, "quantization", None)
    if not qt or 0 not in qt:
        return None
    s = 100.0 * sum(qt[0]) / sum(STD_LUMA)                  # IJG scale factor in percent
    q = (200 - s) / 2 if s <= 100 else 5000 / s
    return float(min(max(q, 1), 100))


def shard_idx(name):
    return int(re.search(r"_(\d+)\.parquet", name).group(1))


def role_of(row):
    n, i = row["image_name"], shard_idx(row["shard"])
    if row["label"] == 1:
        return "fake"
    if re.match(r"^D\d+_", n):
        return "vision"
    if 117 <= i <= 155:
        return "coco"
    if 156 <= i <= 185:
        return "lhq"
    if 93 <= i <= 113:
        return "ffhq"
    return "other"


def select():
    man = json.load(open(os.path.join(ARTIFACTS, "cf_manifest.json")))
    files = [f for v in man["shards"].values() for f in v]
    d = pd.read_parquet(os.path.join(ARTIFACTS, "cf_small_meta.parquet"))
    d = d[d.shard.isin(files)].copy()
    d["pos"] = d["row_in_group"]
    d["role"] = d.apply(role_of, axis=1)
    ev = pd.read_parquet(os.path.join(ARTIFACTS, "cf_eval_meta.parquet"))
    ev_models = set(ev[ev.label == 1].model_name.unique())

    fake = d[d.label == 1]
    excluded = fake[fake.model_name.isin(ev_models)]
    fake = fake[~fake.model_name.isin(ev_models)]
    fake = fake.sample(frac=1, random_state=SEED)
    cap = np.where(fake.subset == "Manual", CAP_MANUAL, CAP_PER_GEN)       # 19 distinct architectures vs SD fine-tunes
    fake = fake[fake.groupby("model_name").cumcount().values < cap]
    parts = [fake]
    for role, n in (("coco", 7000), ("lhq", 3000), ("ffhq", 10 ** 9), ("vision", 10 ** 9)):
        r = d[d.role == role]
        parts.append(r.sample(n=min(n, len(r)), random_state=SEED))
    s = pd.concat(parts, ignore_index=True)
    s["generator"] = s.model_name.where(s.label == 1, "real")
    s["group"] = np.where(s.label == 1, s.model_name,
                          np.where(s.role == "vision", s.image_name.str.extract(r"^(D\d+)_")[0].fillna(""),
                                   s.shard + "/" + s.image_name))
    s["split"] = [split_of(u01(("g:" + g) if lab == 1 or role == "vision" else ("i:" + g)))
                  for g, lab, role in zip(s.group, s.label, s.role)]
    reals, fakes = s[s.label == 0], s[s.label == 1]
    r_jpeg = (reals.format == "JPEG").mean()
    R = r_jpeg + (1 - r_jpeg) * P_PNG_REAL
    fj = (fakes.format == "JPEG").mean()
    p_fake = float(np.clip((R - fj) / (1 - fj), 0, 1))
    s["p_recompress"] = np.where(s.format == "JPEG", 0.0, np.where(s.label == 1, p_fake, P_PNG_REAL))
    s.to_parquet(SEL)
    info = dict(n=len(s), fake=len(fakes), real=len(reals), eval_models=len(ev_models),
                fake_rows_excluded_as_eval_generators=len(excluded), real_jpeg_native=float(r_jpeg),
                target_jpeg_share=float(R), p_fake_recompress=p_fake, p_png_real_recompress=P_PNG_REAL)
    print(json.dumps(info, indent=1))
    print(pd.crosstab([s.role, s.label], s.split))
    json.dump(info, open(os.path.join(ARTIFACTS, "cf_selection_info.json"), "w"), indent=1)


def make_views(data, key, p_recompress, qpool):
    im = Image.open(io.BytesIO(data))
    fmt, q_native = im.format, est_jpeg_q(im)
    w0, h0 = im.size
    im = detector.to_rgb(im)
    rng = np.random.RandomState(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    q_applied = None
    if fmt != "JPEG" and rng.rand() < p_recompress and len(qpool):
        q_applied = int(round(qpool[rng.randint(len(qpool))]))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q_applied)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    W, H = im.size
    up = min(W, H) < VIEW
    imn = im
    if up:
        s = VIEW / min(W, H)
        imn = im.resize((max(VIEW, math.ceil(W * s)), max(VIEW, math.ceil(H * s))), Image.BICUBIC)
    W2, H2 = imn.size
    x0, y0 = 8 * rng.randint(0, (W2 - VIEW) // 8 + 1), 8 * rng.randint(0, (H2 - VIEW) // 8 + 1)
    native = np.asarray(imn.crop((x0, y0, x0 + VIEW, y0 + VIEW)))
    s = VIEW / min(W, H)
    g = im.resize((max(VIEW, round(W * s)), max(VIEW, round(H * s))), Image.BICUBIC)
    gx, gy = (g.size[0] - VIEW) // 2, (g.size[1] - VIEW) // 2
    glob = np.asarray(g.crop((gx, gy, gx + VIEW, gy + VIEW)))
    return native, glob, dict(w0=w0, h0=h0, fmt=fmt, q_native=q_native, q_applied=q_applied, upscaled=up, x0=x0, y0=y0)


def build_shard(idx):
    name = f"data/HFCF_small_{idx}.parquet"
    path = os.path.join(RAW, name)
    out = os.path.join(CACHE, f"s{idx}")
    if os.path.exists(out + "_meta.parquet"):
        print("skip (built)", idx)
        return
    s = pd.read_parquet(SEL)
    s = s[s.shard == name].sort_values("pos").reset_index(drop=True)
    qpool = json.load(open(QPOOL)) if os.path.exists(QPOOL) else []
    os.makedirs(CACHE, exist_ok=True)
    n = len(s)
    nat = np.lib.format.open_memmap(out + "_native.npy", "w+", np.uint8, (n, VIEW, VIEW, 3))
    glo = np.lib.format.open_memmap(out + "_glob.npy", "w+", np.uint8, (n, VIEW, VIEW, 3))
    want = {p: i for i, p in enumerate(s.pos.tolist())}
    rows = []
    nrg = pq.ParquetFile(path).metadata.num_row_groups
    base = 0
    for rg in range(nrg):                                   # byte ranges only: pyarrow would load the whole column chunk
        offs, lens = parquet_stream.image_ranges(path, row_group=rg)
        take = [p - base for p in want if base <= p < base + len(offs)]
        for p, data in parquet_stream.iter_rows(path, offs, lens, take):
            pos = base + p
            i = want[pos]
            try:
                a, g, info = make_views(data, f"{name}:{pos}", float(s.p_recompress[i]), qpool)
            except Exception as e:                         # unreadable image: keep slot, mark invalid
                a = g = np.zeros((VIEW, VIEW, 3), np.uint8)
                info = dict(error=repr(e)[:80])
            nat[i], glo[i] = a, g
            rows.append(dict(i=i, **info))
        base += len(offs)
    nat.flush(), glo.flush()
    m = s.join(pd.DataFrame(rows).set_index("i"))
    m.to_parquet(out + "_meta.parquet")
    nerr = int(m["error"].notna().sum()) if "error" in m else 0
    print("built", idx, n, "images; errors:", nerr, flush=True)
    if s.role.iloc[0] in ("coco", "vision"):               # reference JPEG-quality pool for re-compression
        qs = [q for q in m.q_native.dropna().tolist()]
        prev = json.load(open(QPOOL)) if os.path.exists(QPOOL) else []
        json.dump(prev + qs[:2000], open(QPOOL, "w"))


def downloaded_in_order():
    man = json.load(open(os.path.join(ARTIFACTS, "cf_manifest.json")))
    order = ("coco", "vision", "fake", "manual_extra", "lhq", "ffhq")
    return [shard_idx(f) for r in order for f in man["shards"][r] if os.path.exists(os.path.join(RAW, f))]


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    else:
        for i in (downloaded_in_order() if sys.argv[2] == "all" else [int(a) for a in sys.argv[2:]]):
            build_shard(i)
