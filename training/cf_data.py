"""Access to the bias-controlled Community Forensics cache built by cf_build.py.

Class convention follows the rest of the project: y_real = 1 for real, 0 for fake (Community Forensics' own
`label` column is the opposite: 1 = fake). The model output is the logit of P(real).
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from detector.paths import DATA

CACHE = os.path.join(DATA, "commfor", "cache")


def load_meta(split=None):
    parts = []
    for f in sorted(glob.glob(os.path.join(CACHE, "s*_meta.parquet"))):
        m = pd.read_parquet(f)
        m["sidx"] = int(re.search(r"s(\d+)_meta", f).group(1))
        m["row"] = np.arange(len(m))
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    if "error" in df:
        df = df[df["error"].isna()]
    df["y_real"] = 1 - df["label"].astype(int)
    if split is not None:
        df = df[df.split == split]
    return df.reset_index(drop=True)


class CFDataset(Dataset):
    """Returns (transformed PIL->tensor, y_real float, index). `view` is 'native' or 'glob'."""

    def __init__(self, df, view, tf):
        self.df, self.view, self.tf = df.reset_index(drop=True), view, tf
        self._mm = {}
        self.sidx, self.row, self.y = self.df.sidx.values, self.df.row.values, self.df.y_real.values

    def _arr(self, sidx):
        if sidx not in self._mm:
            self._mm[sidx] = np.load(os.path.join(CACHE, f"s{sidx}_{self.view}.npy"), mmap_mode="r")
        return self._mm[sidx]

    def image(self, i):
        return Image.fromarray(np.array(self._arr(self.sidx[i])[self.row[i]]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.tf(self.image(i)), torch.tensor(float(self.y[i])), i
