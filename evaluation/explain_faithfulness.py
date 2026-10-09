"""Quantify whether Grad-CAM maps are connected to the model's decision. Layer is chosen on the VAL split only.

    python -m evaluation.explain_faithfulness general_native_a [--n 160]

For a class-balanced sample of val tiles (half correctly / half wrongly classified when enough errors exist) and each
candidate layer, with sign = +1 if the model says "real" else -1 (i.e. evidence for the predicted class):
  del_gain : mean over images of  [deletion area(random order) - deletion area(map order)]   > 0  = faithful
  ins_gain : mean over images of  [insertion area(map order) - insertion area(random order)]  > 0  = faithful
  beats_random_frac : share of images where BOTH gains are positive
  rho_mean : Spearman between the map and the map of a weight-randomised network (should be low; sanity check)
Writes artifacts/explain_eval_<name>.json. A positive gain shows the map is predictive of the model's own output; it does
not show that a region is manipulated.
"""
import argparse
import json
import os

import numpy as np
import torch

from training import cf_data
from detector import gradcam as explain
from detector import tiling as general
from detector.paths import ARTIFACTS, MODELS

LAYERS = ("bn2", "blocks.4", "blocks.2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = general.load(os.path.join(MODELS, a.name + ".pt"), dev)
    meta = json.load(open(os.path.join(MODELS, a.name + ".meta.json")))
    val = cf_data.load_meta("val").sample(frac=1, random_state=3).reset_index(drop=True).iloc[:1500]
    ds = cf_data.CFDataset(val, ck["view"], None)
    ims = [ds.image(i) for i in range(len(ds))]
    z = np.concatenate([general.tile_logits(model, ims[i:i + 64], dev, flip=False) for i in range(0, len(ims), 64)])
    pred_real = z / meta["temperature"] > 0
    correct = pred_real == (val.y_real.values == 1)
    rng = np.random.RandomState(0)
    ok, bad = np.where(correct)[0], np.where(~correct)[0]
    pick = {"correct": rng.choice(ok, min(a.n // 2, len(ok)), replace=False),
            "wrong": rng.choice(bad, min(a.n // 2, len(bad)), replace=False)}
    out = {"name": a.name, "layers": {}}
    mods = dict(model.named_modules())
    for layer in LAYERS:
        res = {}
        for grp, idx in pick.items():
            dg, ig, rho = [], [], []
            for i in idx:
                x = general._TF(ims[i]).unsqueeze(0).to(dev)
                sign = 1.0 if pred_real[i] else -1.0
                cam, _ = explain.gradcam(model, x, mods[layer], sign)
                c = explain.deletion_insertion(model, x, cam, sign)
                A = explain.curve_area
                dg.append(A(c["deletion_random"]) - A(c["deletion_map"]))
                ig.append(A(c["insertion_map"]) - A(c["insertion_random"]))
                rho.append(explain.randomization_check(model, x, layer, sign))
            dg, ig = np.array(dg), np.array(ig)
            res[grp] = dict(n=int(len(idx)), del_gain=float(dg.mean()), ins_gain=float(ig.mean()),
                            beats_random_frac=float(((dg > 0) & (ig > 0)).mean()), rho_mean=float(np.nanmean(rho)))
        out["layers"][layer] = res
        print(layer, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(ARTIFACTS, f"explain_eval_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
