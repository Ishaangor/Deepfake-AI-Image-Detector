"""Scan Community Forensics parquet shards for METADATA ONLY (no image bytes are downloaded).

    python cf_scan.py small   -> artifacts/cf_small_meta.parquet
    python cf_scan.py eval    -> artifacts/cf_eval_meta.parquet

For each image it records shard, row_group, row-in-group plus the label/generator/source/format/resolution columns, so
a stratified subset can later be fetched with parquet row-group range reads (cf_fetch.py).
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPOS = {"small": "OwensLab/CommunityForensics-Small", "eval": "OwensLab/CommunityForensics-Eval"}
COLS = ["image_name", "format", "resolution", "mode", "model_name", "real_source", "subset", "split", "label",
        "architecture"]


def scan(repo, path, fs):
    for attempt in range(4):
        try:
            with fs.open(f"datasets/{repo}/{path}", "rb") as f:
                pf = pq.ParquetFile(f)
                parts, off = [], 0
                for g in range(pf.num_row_groups):
                    t = pf.read_row_group(g, columns=COLS).to_pandas()
                    t["row_group"], t["row_in_group"], t["shard"] = g, range(len(t)), path
                    parts.append(t)
                return pd.concat(parts)
        except Exception as e:  # transient network errors
            err = e
    print("FAILED", path, err, flush=True)
    return None


def main(which):
    repo = REPOS[which]
    fs = HfFileSystem()
    files = sorted(s.rfilename for s in HfApi().dataset_info(repo, files_metadata=False).siblings
                   if s.rfilename.endswith(".parquet"))
    print(len(files), "shards", flush=True)
    out = []
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(lambda p: scan(repo, p, fs), files)):
            if r is not None:
                out.append(r)
            if i % 10 == 0:
                print(i, sum(len(o) for o in out), flush=True)
    df = pd.concat(out, ignore_index=True)
    df["resolution"] = df["resolution"].map(lambda r: f"{r[0]}x{r[1]}" if r is not None else None)
    df.to_parquet(f"artifacts/cf_{which}_meta.parquet")
    print(df.shape)


if __name__ == "__main__":
    main(sys.argv[1])
