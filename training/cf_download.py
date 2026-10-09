"""Download the selected Community Forensics-Small parquet shards (resumable, 3 parallel streams).

    python -m training.cf_download            # writes data/commfor/raw/data/<shard>.parquet and artifacts/cf_manifest.json

Shard choice is fixed by seed BEFORE any image is looked at. Roles:
  fake   : 9 random shards of the 93 mixed-generator fake shards (idx 0-92)          ~1.2 GB each
  coco   : 3 random COCO real shards (idx 117-154, 640x480 JPEG)                     ~0.15 GB each
  lhq    : 2 random landscape real shards (idx 156-185, 768x512 PNG)                 ~1.8 GB each
  ffhq   : 1 random FFHQ real shard (idx 93-113, 1024x1024 PNG, faces)               ~4.1 GB
  vision : shards 115, 116 (VISION phone photos, native JPEG)                        ~2.6 / 1.1 GB
License: CC BY-NC-SA 4.0 (non-commercial research). Raw files are not tracked by git (data/commfor/ is ignored).
"""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download
from detector.paths import ARTIFACTS, DATA

REPO = "OwensLab/CommunityForensics-Small"
OUT = os.path.join(DATA, "commfor", "raw")


def manifest(seed=20261009):
    rng = random.Random(seed)
    pick = lambda lo, hi, k: sorted(rng.sample(range(lo, hi + 1), k))
    roles = {"fake": pick(0, 92, 9), "coco": pick(117, 154, 3), "lhq": pick(156, 185, 2),
             "ffhq": pick(93, 113, 1), "vision": [115, 116],
             # greedy cover of the 19 hand-collected 'Manual' architectures (GANs, DiT, guided diffusion, ...) that the
             # random fake draw missed; chosen from shard metadata only, before looking at any image
             "manual_extra": [74, 76, 66, 79, 67, 70, 87]}
    return {"repo": REPO, "seed": seed,
            "shards": {role: [f"data/HFCF_small_{i}.parquet" for i in idx] for role, idx in roles.items()}}


def fetch(fname):
    for attempt in range(6):
        try:
            p = hf_hub_download(REPO, fname, repo_type="dataset", local_dir=OUT)
            print("DONE", fname, round(os.path.getsize(p) / 1e9, 2), "GB", flush=True)
            return p
        except Exception as e:
            print("retry", fname, attempt, repr(e)[:120], flush=True)
    raise RuntimeError(fname)


if __name__ == "__main__":
    os.makedirs(ARTIFACTS, exist_ok=True)
    m = manifest()
    json.dump(m, open(os.path.join(ARTIFACTS, "cf_manifest.json"), "w"), indent=1)
    order = ("manual_extra",) if "extra" in os.sys.argv[1:] else ("coco", "vision", "fake", "lhq", "ffhq")
    files = [f for role in order for f in m["shards"][role]]
    print(len(files), "shards:", m["shards"], flush=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(fetch, files))
    print("ALL DOWNLOADS COMPLETE", flush=True)
