"""Score the independent CommunityForensics-Eval (CompEval) subset with a general checkpoint. EVALUATION ONLY.

    python eval_general.py general_native_a [--bench v1|v2] [--conds clean,jpeg75,jpeg50,half,quarter,blur1,gray,social] [--limit N]

Writes artifacts/evalraw_<name>.parquet (one row per image x condition: logit of P(real), tile spread, tile count).
Conditions are applied to the NATIVE-resolution image before tiling/resizing and were fixed in advance; they are never
used for model selection (selection uses the 'val' split in train_general.py).
Metrics are computed by eval_metrics.py from the raw scores.
"""
import argparse
import io
import os
import time

import pandas as pd
import torch
from PIL import Image, ImageFilter

import detector
import general

EVAL_DIR = os.path.join("data", "commfor", "eval")


def _jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _scale(im, f):
    return im.resize((max(8, round(im.width * f)), max(8, round(im.height * f))), Image.BILINEAR)


def _social(im):                                                # downscale long side to 1080 (if larger) + JPEG q70
    s = 1080 / max(im.size)
    return _jpeg(_scale(im, s) if s < 1 else im, 70)


CONDS = {
    "clean": lambda im: im,
    "jpeg75": lambda im: _jpeg(im, 75),
    "jpeg50": lambda im: _jpeg(im, 50),
    "half": lambda im: _scale(im, 0.5),
    "quarter": lambda im: _scale(im, 0.25),
    "blur1": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
    "gray": lambda im: im.convert("L").convert("RGB"),
    "social": _social,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench", default="v1", choices=["v1", "v2"],
                    help="v1 = retired CompEval sample (zero-shot result of Small-only models); "
                         "v2 = held-out generator families + held-out real row groups (ce_fetch.py role 'bench')")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = general.load(os.path.join("models", a.name + ".pt"), dev)
    if a.bench == "v1":
        sel, root = pd.read_parquet(os.path.join("artifacts", "cf_eval_sel.parquet")), EVAL_DIR
    else:
        sel, root = pd.read_parquet(os.path.join("artifacts", "ce_sel.parquet")), os.path.join("data", "commfor", "ce")
        sel = sel[sel.role == "bench"]
    sel = sel[[os.path.exists(os.path.join(root, f)) for f in sel.file]]
    if a.limit:
        sel = sel.sample(n=min(a.limit, len(sel)), random_state=0)
    conds = a.conds.split(",")
    rows, t0 = [], time.time()
    for k, f in enumerate(sel.file):
        try:
            im = detector.load_rgb(os.path.join(root, f))
        except Exception as e:
            rows.append(dict(file=f, cond="clean", error=repr(e)[:80]))
            continue
        for c in conds:
            try:
                r = general.score(model, ck, CONDS[c](im), dev)
                rows.append(dict(file=f, cond=c, logit=r["logit"], std=r["std"], n_tiles=r["n_tiles"], w=im.width, h=im.height))
            except Exception as e:
                rows.append(dict(file=f, cond=c, error=repr(e)[:80]))
        if k % 100 == 0:
            print(f"{k}/{len(sel)} ({time.time() - t0:.0f}s)", flush=True)
    out = os.path.join("artifacts", f"evalraw_{a.name}" + ("" if a.bench == "v1" else "__v2") + ".parquet")
    pd.DataFrame(rows).to_parquet(out)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
