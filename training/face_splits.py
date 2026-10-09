"""Leak-aware, cluster-grouped splits (v2).

Why: the original split (face_finetune.py) is random per file, so exact / near-duplicate images can
sit on both sides of a split, and 'rvf10k' is an exact copy of 'deepfake_dataset'. Here:

  * pool = the ORIGINAL train / val / test lists (rvf10k is NOT used: it is a verified duplicate)
  * clusters = exact (md5) + near-duplicate (256-bit dHash, Hamming <= NEAR_THR) connected components
  * a cluster that touches train keeps its train members and LOSES its val/test members, so val, calib
    and test never contain a near-copy of a training image (the original checkpoint was trained on the
    same train set, so comparisons stay fair and the old test set stays untouched by both models)
  * the cleaned old val set is split by cluster into val_sel (checkpoint / augmentation selection) and
    calib (temperature scaling only); test is the cleaned old test set and is never used for selection

Identity: no identity metadata exists (FFHQ ids are not person ids, StyleGAN faces have no identity), so
identity leakage is NOT checked here and must not be claimed as eliminated.

Run:  python -m training.face_splits     -> artifacts/splits_v2.json (+ artifacts/hash_cache.json)
"""
import hashlib
import io
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from detector.paths import ARTIFACTS

ART = ARTIFACTS
CACHE = os.path.join(ART, "hash_cache.json")
OUT = os.path.join(ART, "splits_v2.json")
NEAR_THR = 10          # Hamming bits out of 256
SEED = 42


def subset_of(path):
    p = path.replace("\\", "/")
    return "ai_faces" if "/ai_faces/" in p or p.startswith("data/ai_faces") else "dfd"


def source_of(path, y):
    """Provenance as far as the metadata allows (see data/*.csv original_path for the dfd subset)."""
    if subset_of(path) == "dfd":
        return "FFHQ(real)" if y == 1 else "1M-fake-faces(StyleGAN)"
    return "ai_faces/real(source unknown)" if y == 1 else "ai_faces/AI(generator unknown)"


def _dhash(im, n=16):
    g = im.convert("L").resize((n + 1, n), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return "".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).flatten())


def hashes_for(paths):
    cache = json.load(open(CACHE)) if os.path.isfile(CACHE) else {}
    new = 0
    for p in paths:
        if p in cache:
            continue
        with open(p, "rb") as f:
            b = f.read()
        im = Image.open(io.BytesIO(b))
        w, h = im.size
        im.draft("RGB", (128, 128))
        cache[p] = {"md5": hashlib.md5(b).hexdigest(), "dh": _dhash(im), "w": w, "h": h}
        new += 1
        if new % 4000 == 0:
            print(f"  hashed {new} new images", flush=True)
    if new:
        json.dump(cache, open(CACHE, "w"))
    return cache


def clusters(paths, cache):
    n = len(paths)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_md5 = defaultdict(list)
    for i, p in enumerate(paths):
        by_md5[cache[p]["md5"]].append(i)
    for g in by_md5.values():
        for j in g[1:]:
            union(g[0], j)
    bits = np.array([[int(c) for c in cache[p]["dh"]] for p in paths], dtype=np.uint8)
    packed = np.packbits(bits, axis=1)
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for s in range(0, n, 64):
        d = pop[np.bitwise_xor(packed[s:s + 64, None, :], packed[None, :, :])].sum(2)
        ii, jj = np.where(d <= NEAR_THR)
        for a, b in zip(ii + s, jj):
            if a < b:
                union(int(a), int(b))
    return [find(i) for i in range(n)]


def build():
    import train_finetune as tfn   # original lists only; importing does not train anything
    os.makedirs(ART, exist_ok=True)
    orig = {"train": tfn.train, "val": tfn.val, "test": tfn.test}
    rows = [(sp, p, y) for sp, items in orig.items() for p, y in items]
    paths = [p for _, p, _ in rows]
    print(f"hashing {len(paths)} images (cached in {CACHE}) ...", flush=True)
    cache = hashes_for(paths)
    cid = clusters(paths, cache)
    members = defaultdict(list)
    for i, c in enumerate(cid):
        members[c].append(i)

    keep = {"train": [], "val": [], "test": []}
    dropped = Counter()
    seen_md5 = {"train": set(), "test": set()}
    for c, idx in members.items():
        sps = {rows[i][0] for i in idx}
        target = "train" if "train" in sps else ("test" if "test" in sps else "val")
        for i in idx:
            sp, p, y = rows[i]
            if sp != target:
                dropped[f"{sp}_has_near_copy_in_{target}"] += 1
                continue
            m = cache[p]["md5"]
            if target in seen_md5:
                if m in seen_md5[target]:
                    dropped[f"{sp}_exact_duplicate_removed"] += 1
                    continue
                seen_md5[target].add(m)
            keep[target].append((p, y, c))

    # cluster-grouped split of the cleaned old-val into val_sel / calib, stratified by subset x class
    rng = random.Random(SEED)
    strata = defaultdict(lambda: defaultdict(list))
    for p, y, c in keep["val"]:
        strata[(subset_of(p), y)][c].append((p, y))
    val_sel, calib = [], []
    for key in sorted(strata):
        cs = sorted(strata[key])
        rng.shuffle(cs)
        half = len(cs) // 2
        for c in cs[:half]:
            val_sel += strata[key][c]
        for c in cs[half:]:
            calib += strata[key][c]

    splits = {"train": [(p, y) for p, y, _ in keep["train"]], "val_sel": val_sel, "calib": calib,
              "test": [(p, y) for p, y, _ in keep["test"]]}
    meta = {"near_threshold_bits_of_256": NEAR_THR, "seed": SEED, "dropped": dict(dropped),
            "counts": {k: len(v) for k, v in splits.items()}}
    for k, v in splits.items():
        meta[f"{k}_by_source"] = dict(Counter(f"{source_of(p, y)}|{'real' if y else 'fake'}" for p, y in v))
    # sanity: no cluster shared across splits
    idx_of = {q: i for i, q in enumerate(paths)}
    owner = defaultdict(set)
    for k, v in splits.items():
        for p, _ in v:
            owner[cid[idx_of[p]]].add(k)
    meta["clusters_spanning_splits"] = sum(1 for s in owner.values() if len(s) > 1)
    json.dump({"meta": meta, "splits": splits}, open(OUT, "w"))
    print(json.dumps(meta, indent=1))


def load():
    d = json.load(open(OUT))
    return {k: [tuple(x) for x in v] for k, v in d["splits"].items()}, d["meta"]


if __name__ == "__main__":
    build()
