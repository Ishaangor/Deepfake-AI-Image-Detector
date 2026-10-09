"""Build cache shard 910 from the CompEval-derived TRAIN / VAL / CALIB pool fetched by ce_fetch.py.

    python ce_build.py

Rows with role == 'bench' (the new independent benchmark CompEval-v2) are NEVER put in the cache.
Shortcut control is the same as cf_build: PNG images are JPEG-recompressed with probability p (fakes 0.9, reals 0.75) at
a quality drawn from the empirical qualities of natively-JPEG real images (artifacts/cf_qpool.json), so the JPEG share is
about equal in both classes (Firefly fakes and most LAION reals are natively JPEG).
Splits: fakes by generator-family role from ce_fetch (train / val); reals by row-group role (train / val / calib).
"""
import json
import os

import numpy as np
import pandas as pd

import cf_build

CE = os.path.join("data", "commfor", "ce")
IDX = 910
P_FAKE, P_REAL = 0.9, cf_build.P_PNG_REAL


def main():
    s = pd.read_parquet(os.path.join("artifacts", "ce_sel.parquet"))
    s = s[s.role != "bench"].reset_index(drop=True)
    s = s[[os.path.exists(os.path.join(CE, f)) for f in s.file]].reset_index(drop=True)
    qpool = json.load(open(cf_build.QPOOL))
    out = os.path.join(cf_build.CACHE, f"s{IDX}")
    n = len(s)
    nat = np.lib.format.open_memmap(out + "_native.npy", "w+", np.uint8, (n, cf_build.VIEW, cf_build.VIEW, 3))
    glo = np.lib.format.open_memmap(out + "_glob.npy", "w+", np.uint8, (n, cf_build.VIEW, cf_build.VIEW, 3))
    rows = []
    for k, r in enumerate(s.itertuples()):
        fake = r.label == 1
        base = dict(image_name=r.file, label=int(r.label), role="ce_fake" if fake else "ce_" + str(r.real_source).lower(),
                    generator=r.model_name if fake else "real", model_name=r.model_name if fake else None,
                    group=r.family if fake else f"{r.shard}/{r.row_group}", shard="ce", pos=k, split=r.role)
        try:
            data = open(os.path.join(CE, r.file), "rb").read()
            p = 0.0 if r.format == "JPEG" else (P_FAKE if fake else P_REAL)
            a, g, info = cf_build.make_views(data, "ce:" + r.file, p, qpool)
            nat[k], glo[k] = a, g
            rows.append(dict(base, p_recompress=p, **info))
        except Exception as e:
            nat[k] = glo[k] = 0
            rows.append(dict(base, error=repr(e)[:80]))
    nat.flush(), glo.flush()
    m = pd.DataFrame(rows)
    m.to_parquet(out + "_meta.parquet")
    print(pd.crosstab(m.role, m.split).to_string(), "\nerrors:", int(m["error"].notna().sum()) if "error" in m else 0)


if __name__ == "__main__":
    main()
