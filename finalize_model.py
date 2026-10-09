"""Calibration + crop-policy selection for one checkpoint, using ONLY the 'calib' and 'val_sel' splits.

    python finalize_model.py candidate_B_full.pt                 # fits temperature on calib
    python finalize_model.py best_finetuned.pt --keep-temperature # keeps the shipped T (original checkpoint)

Writes models/<name>.meta.json read by detector.py. The test split is never touched.

Pre-registered rules (fixed before looking at any test result):
  * temperature = argmin NLL on calib over the SEEN perturbation conditions with equal weight per condition
    ("mix"); the clean-only temperature is stored too. The mix is the primary because deployment images are
    not all clean.
  * crop_policy = "face" only if the Haar face crop beats the full image by > 0.5 pp pooled accuracy on
    val_sel (clean images); otherwise "none".
"""
import json
import os
import sys

import numpy as np
import timm
import torch
from scipy.optimize import minimize_scalar

import detector
import perturb
import splits
from train_robust import _to_tensor


def get_logits(model, imgs, dev):
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([_to_tensor(im) for im in imgs[i:i + 64]]).to(dev)
        with torch.no_grad():
            out.append(((model(x) + model(torch.flip(x, [3]))) / 2).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


def nll(z, y, t):
    p = 1 / (1 + np.exp(-z / t))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_t(z, y):
    return float(minimize_scalar(lambda t: nll(z, y, t), bounds=(0.2, 5.0), method="bounded").x)


def main(name, keep_t):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join("models", name)
    ck = torch.load(path, map_location=dev)
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=1)
    model.load_state_dict(ck["state"])
    model.to(dev).eval()
    sp, _ = splits.load()

    calib = [(detector.load_rgb(p), y) for p, y in sp["calib"]]
    yc = np.array([y for _, y in calib], dtype=np.float64)
    per_cond = {c: get_logits(model, [fn(im) for im, _ in calib], dev) for c, fn in perturb.SEEN.items()}
    t_clean = fit_t(per_cond["clean"], yc)
    t_mix = fit_t(np.concatenate(list(per_cond.values())), np.tile(yc, len(per_cond)))

    val = [(detector.load_rgb(p), y) for p, y in sp["val_sel"]]
    yv = np.array([y for _, y in val])
    boxes = [detector.find_face(im) for im, _ in val]
    regions = [im.crop(b) if b else im for (im, _), b in zip(val, boxes)]
    acc_none = float(((get_logits(model, [perturb._base(im) for im, _ in val], dev) >= 0) == yv).mean())
    acc_face = float(((get_logits(model, [perturb._base(r) for r in regions], dev) >= 0) == yv).mean())
    policy = "face" if acc_face > acc_none + 0.005 else "none"

    side = os.path.splitext(path)[0] + ".meta.json"
    old_t = None
    if keep_t:
        old_t = json.load(open(os.path.join("models", "calibration.json")))["temperature"]
    meta = {"temperature": old_t if keep_t else t_mix,
            "temperature_calib_mix": t_mix, "temperature_calib_clean_only": t_clean,
            "temperature_source": "shipped value kept (fit on the OLD validation set, not a separate split)" if keep_t
            else f"fit on calib split (n={len(calib)}) over {list(perturb.SEEN)}",
            "crop_policy": policy,
            "crop_selection": {"split": "val_sel", "n": len(val), "acc_none": acc_none, "acc_face": acc_face,
                               "face_found_frac": float(np.mean([b is not None for b in boxes])),
                               "rule": "face only if acc_face > acc_none + 0.005"},
            "arch": ck["arch"], "img": ck["img"], "checkpoint": name,
            "checkpoint_info": {k: ck[k] for k in ("epoch", "val_sel_composite", "variant", "init") if k in ck}}
    json.dump(meta, open(side, "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], "--keep-temperature" in sys.argv)
