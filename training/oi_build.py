"""Add web-style REAL photographs from Open Images (validation split) to the training cache as shard 900.

    python -m training.oi_build [N=6000]

Why: Community Forensics-Small's reals are only COCO / VISION phone photos / LHQ landscapes / FFHQ faces, while the
independent CompEval benchmark's biggest real source is web photographs (LAION). A detector trained without web-style
reals can learn "web-photo look = fake". Open Images images are Flickr photos; the CSV lists a licence per image and we
keep only CC BY (2.0) ones. N images are sampled with a fixed seed and downloaded from the public CVDF S3 mirror.
Native JPEGs are kept untouched (p_recompress = 0). Splits are by image hash 80/10/10 like the other real sources.
Raw files: data/openimages/raw (git-ignored). Output: data/commfor/cache/s900_{native,glob}.npy + s900_meta.parquet.
"""
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from training import cf_build
from detector.paths import DATA

CSV = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
MIRROR = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
RAW = os.path.join(DATA, "openimages", "raw")
IDX = 900


def get(i):
    p = os.path.join(RAW, i + ".jpg")
    if os.path.exists(p):
        return i
    for _ in range(3):
        try:
            open(p, "wb").write(urllib.request.urlopen(MIRROR.format(i), timeout=30).read())
            return i
        except Exception:
            pass
    return None


def main(n):
    os.makedirs(RAW, exist_ok=True)
    csv = os.path.join(DATA, "openimages", "validation.csv")
    if not os.path.exists(csv):
        urllib.request.urlretrieve(CSV, csv)
    d = pd.read_csv(csv)
    lic = d.License.astype(str)
    d = d[lic.str.contains("by/2.0", na=False) & ~lic.str.contains("nc|nd|sa", na=False)]
    print(len(d), "CC-BY images available;", d.License.iloc[0], flush=True)
    ids = d.sample(n=min(n, len(d)), random_state=cf_build.SEED).ImageID.tolist()
    with ThreadPoolExecutor(32) as ex:
        ok = [i for i in ex.map(get, ids) if i]
    print("downloaded", len(ok), flush=True)
    out = os.path.join(cf_build.CACHE, f"s{IDX}")
    nat = np.lib.format.open_memmap(out + "_native.npy", "w+", np.uint8, (len(ok), cf_build.VIEW, cf_build.VIEW, 3))
    glo = np.lib.format.open_memmap(out + "_glob.npy", "w+", np.uint8, (len(ok), cf_build.VIEW, cf_build.VIEW, 3))
    rows = []
    for k, i in enumerate(ok):
        try:
            a, g, info = cf_build.make_views(open(os.path.join(RAW, i + ".jpg"), "rb").read(), f"oi:{i}", 0.0, [])
            nat[k], glo[k] = a, g
            rows.append(dict(image_name=i, label=0, role="openimages", generator="real", group="oi/" + i, shard="openimages",
                             pos=k, p_recompress=0.0, split=cf_build.split_of(cf_build.u01("i:oi/" + i)), **info))
        except Exception as e:
            nat[k] = glo[k] = 0
            rows.append(dict(image_name=i, label=0, role="openimages", pos=k, error=repr(e)[:80]))
    nat.flush(), glo.flush()
    m = pd.DataFrame(rows)
    m.to_parquet(out + "_meta.parquet")
    print(m.split.value_counts().to_dict(), "errors:", int(m["error"].notna().sum()) if "error" in m else 0)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
