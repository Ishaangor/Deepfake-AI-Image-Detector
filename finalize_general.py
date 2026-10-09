"""Fit the temperature of a general checkpoint on the 'calib' split (generators and cameras unseen in training).

    python finalize_general.py general_native_a

* temperature = argmin NLL over calib, equal weight per SEEN condition (clean, jpeg50, down128, blur1.0, gray), computed
  with the same single-view + flip-TTA pipeline as validation. Neither 'train', the val selection data nor CompEval is used.
* The decision threshold stays at calibrated P(real) = 0.5 (logit 0).
Writes models/<name>.meta.json (read by detector.py).
"""
import json
import os
import sys

import numpy as np
import torch

import cf_data
import general
import perturb
from finalize_model import fit_t, nll

CONDS = ("clean", "jpeg50", "down128", "blur1.0", "gray")
TARGET_PRECISION = 0.90
EXPLAIN_LAYER = "blocks.4"        # chosen by explain_eval.py on the val split (most faithful of bn2 / blocks.4 / blocks.2)


def decision_band(p_real, y_real, target=TARGET_PRECISION):
    """[lo, hi] on calibrated P(real): below lo -> 'likely fake', above hi -> 'likely real', between -> 'inconclusive'.

    hi = smallest threshold whose 'real' verdicts reach `target` precision, lo = largest threshold whose 'fake' verdicts
    reach it, with classes weighted equally (calibration data are not deployment-prevalence data). If a side never reaches
    the target it is closed off (hi = 1, lo = 0). An empty/inverted band falls back to the 0.5 threshold."""
    w = np.where(y_real == 1, 0.5 / y_real.sum(), 0.5 / (1 - y_real).sum())
    order = np.argsort(p_real)
    p, yr, ww = p_real[order], y_real[order], w[order]
    real_w, fake_w = np.cumsum(ww * yr), np.cumsum(ww * (1 - yr))
    tot_real, tot_fake = real_w[-1], fake_w[-1]
    hi, lo = 1.0, 0.0
    for i in range(len(p)):                                           # real verdicts: all p >= p[i]
        r, f = tot_real - (real_w[i - 1] if i else 0.0), tot_fake - (fake_w[i - 1] if i else 0.0)
        if r / max(r + f, 1e-12) >= target:
            hi = float(p[i])
            break
    for i in range(len(p) - 1, -1, -1):                               # fake verdicts: all p <= p[i]
        if fake_w[i] / max(fake_w[i] + real_w[i], 1e-12) >= target:
            lo = float(p[i])
            break
    return [lo, hi] if lo < hi else [0.5, 0.5]


@torch.no_grad()
def logits(model, ds, fn, dev, bs=64):
    out = []
    for i in range(0, len(ds), bs):
        ts = [fn(ds.image(j)) for j in range(i, min(i + bs, len(ds)))]
        out.append(general.tile_logits(model, ts, dev, flip=True))
    return np.concatenate(out)


def main(name):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = general.load(os.path.join("models", name + ".pt"), dev)
    calib = cf_data.load_meta("calib")
    ds = cf_data.CFDataset(calib, ck["view"], None)
    y = ds.y.astype(np.float64)
    zs = {c: logits(model, ds, perturb.SEEN[c], dev) for c in CONDS}
    z, yy = np.concatenate(list(zs.values())), np.tile(y, len(CONDS))
    t = fit_t(z, yy)
    t_clean = fit_t(zs["clean"], y)
    band = decision_band(1 / (1 + np.exp(-z / t)), yy)
    meta = dict(name=name, view=ck["view"], arch=ck["arch"], temperature=t, temperature_clean_only=t_clean,
                calib_n=int(len(y)), calib_generators=int(calib[calib.y_real == 0].model_name.nunique()),
                nll_before=nll(z, yy, 1.0), nll_after=nll(z, yy, t), conditions=list(CONDS),
                val_composite=ck.get("val_composite"), epoch=ck.get("epoch"), band=band, band_target_precision=TARGET_PRECISION,
                explain_layer=EXPLAIN_LAYER)
    json.dump(meta, open(os.path.join("models", name + ".meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
