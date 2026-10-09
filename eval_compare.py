"""Head-to-head evaluation of the original checkpoint vs. robustness candidates on the untouched test split.

    python eval_compare.py

Nothing here selects anything: temperatures and crop policies come from models/<name>.meta.json (fitted on
calib / val_sel). All models see exactly the same test images and the same deterministic perturbations.
Writes artifacts/eval_report.json and prints markdown tables.
"""
import glob
import json
import os

import numpy as np
import timm
import torch
from sklearn.metrics import roc_auc_score

import detector
import perturb
import splits
from finalize_model import get_logits

B = 1000
ORIG = "best_finetuned.pt"


def sig(z):
    return 1 / (1 + np.exp(-z))


def ece(p, y, nb=10):
    conf = np.where(p >= .5, p, 1 - p)
    cor = ((p >= .5).astype(int) == y).astype(float)
    e = 0.0
    edges = np.linspace(.5, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == .5 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(e)


def core(p, y):
    pred = (p >= .5).astype(int)
    tp_f = ((pred == 0) & (y == 0)).sum(); fn_f = ((pred == 1) & (y == 0)).sum(); fp_f = ((pred == 0) & (y == 1)).sum()
    fr = tp_f / max(tp_f + fn_f, 1); fpr = tp_f / max(tp_f + fp_f, 1)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    return dict(acc=float((pred == y).mean()), fake_recall=float(fr), fake_precision=float(fpr),
                fake_f1=float(2 * fr * fpr / max(fr + fpr, 1e-12)), auc=float(auc),
                brier=float(np.mean((p - y) ** 2)), ece=ece(p, y))


def full(p, y):
    pred = (p >= .5).astype(int)
    d = core(p, y)
    conf = np.where(p >= .5, p, 1 - p)
    hi = conf >= .9
    d.update(n=int(len(y)), n_fake=int((y == 0).sum()), n_real=int((y == 1).sum()),
             real_recall=float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
             real_precision=float((y[pred == 1] == 1).mean()) if (pred == 1).any() else float("nan"),
             confusion=[[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)],
             mean_conf=float(conf.mean()), frac_conf_ge90=float(hi.mean()),
             acc_when_conf_ge90=float((pred[hi] == y[hi]).mean()) if hi.any() else float("nan"),
             errors_conf_ge90=int((pred[hi] != y[hi]).sum()))
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp, meta = splits.load()
    hc = json.load(open(splits.CACHE))
    test = sp["test"]
    y = np.array([t[1] for t in test])
    imgs = [detector.load_rgb(p) for p, _ in test]
    print(f"test n={len(test)} fake={int((y == 0).sum())} real={int((y == 1).sum())}", flush=True)

    names = [ORIG] + sorted(os.path.basename(p) for p in glob.glob("models/candidate_*.pt")
                            if os.path.isfile(os.path.splitext(p)[0] + ".meta.json"))
    models, T = {}, {}
    for n in names:
        ck = torch.load(os.path.join("models", n), map_location=dev)
        m = timm.create_model(ck["arch"], pretrained=False, num_classes=1)
        m.load_state_dict(ck["state"]); m.to(dev).eval()
        models[n] = m
        T[n] = detector._meta(os.path.join("models", n), ck)
    labels = {ORIG: "original (shipped T)"}
    for n in names[1:]:
        labels[n] = n.replace("candidate_", "cand ").replace(".pt", "")
    extra_orig = "original (T refit on calib)"          # same logits, calib-fitted temperature
    meta_o = json.load(open("models/best_finetuned.meta.json"))
    t_refit = meta_o["temperature_calib_mix"]
    print("crop policies:", {labels[n]: T[n]["crop_policy"] for n in names}, flush=True)

    # ---- logits for every perturbation condition (+ crop variant), same images for every model
    Z = {n: {} for n in names}
    for cond, fn in perturb.ALL.items():
        dg = [fn(im) for im in imgs]
        for n in names:
            Z[n][cond] = get_logits(models[n], dg, dev)
        print(f"  condition {cond} done", flush=True)
    boxes = [detector.find_face(im) for im in imgs]
    reg = [perturb._base(im.crop(b) if b else im) for im, b in zip(imgs, boxes)]
    for n in names:
        Z[n]["clean_face_crop"] = get_logits(models[n], reg, dev)

    def P(n, cond, temp=None):
        return sig(Z[n][cond] / (T[n]["temperature"] if temp is None else temp))

    # row label -> (checkpoint name, temperature override)
    spec = {labels[n]: (n, None) for n in names}
    spec[extra_orig] = (ORIG, t_refit)
    rows = {k: (lambda n=n, t=t: P(n, "clean", t)) for k, (n, t) in spec.items()}
    pol = {labels[n]: (lambda n=n: P(n, "clean_face_crop" if T[n]["crop_policy"] == "face" else "clean")) for n in names}
    report = {"test_counts": {"n": len(y), "fake": int((y == 0).sum()), "real": int((y == 1).sum())},
              "split_meta": meta, "temperatures": {labels[n]: T[n]["temperature"] for n in names},
              "temperature_original_refit_on_calib": t_refit,
              "crop_policy": {labels[n]: T[n]["crop_policy"] for n in names}}

    # ---- sanity: this evaluation path must equal detector.analyze (crop none)
    for n in names:
        for i in (0, len(test) // 2, len(test) - 1):
            a = detector.analyze(test[i][0], crop="none", model=n).real_prob
            assert abs(a - float(P(n, "clean")[i])) < 2e-3, (n, i, a, float(P(n, "clean")[i]))
    print("parity with detector.analyze(crop='none'): OK", flush=True)

    # ---- 1. headline metrics + bootstrap CIs
    def headline(pf, tag):
        out = {}
        pm = {k: f() for k, f in pf.items()}
        rng = np.random.RandomState(0)
        idxs = [rng.randint(0, len(y), len(y)) for _ in range(B)]
        for k, p in pm.items():
            fm = full(p, y)
            boots = [core(p[ix], y[ix]) for ix in idxs]
            fm["ci95"] = {m: [float(np.nanpercentile([b[m] for b in boots], 2.5)),
                              float(np.nanpercentile([b[m] for b in boots], 97.5))] for m in boots[0]}
            out[k] = fm
        base = labels[ORIG]
        for k, p in pm.items():
            if k == base:
                continue
            bd = [(core(p[ix], y[ix]), core(pm[base][ix], y[ix])) for ix in idxs]
            out[k]["paired_diff_vs_original"] = {
                m: {"mean": float(np.nanmean([a[m] - b[m] for a, b in bd])),
                    "ci95": [float(np.nanpercentile([a[m] - b[m] for a, b in bd], 2.5)),
                             float(np.nanpercentile([a[m] - b[m] for a, b in bd], 97.5))],
                    "p_better": float(np.mean([(a[m] < b[m]) if m in ("brier", "ece") else (a[m] > b[m])
                                               for a, b in bd]))}
                for m in ("acc", "fake_recall", "fake_precision", "auc", "brier", "ece")}
        report[tag] = out
        return out
    h = headline(rows, "headline_full_image")
    hp = headline(pol, "headline_own_crop_policy")

    # ---- 2. by source dataset and by native resolution (clean, full image)
    src = np.array([splits.source_of(p, yy) for p, yy in test])
    short = np.array([min(hc[p]["w"], hc[p]["h"]) for p, _ in test])
    res_bin = np.where(short < 200, "<200px", np.where(short < 300, "200-299px", ">=300px"))
    by = {"source": {}, "resolution": {}}
    for kind, arr in (("source", src), ("resolution", res_bin)):
        for v in sorted(set(arr)):
            m = arr == v
            by[kind][v] = {"n": int(m.sum()), "classes": {"fake": int((y[m] == 0).sum()), "real": int((y[m] == 1).sum())}}
            for k, f in rows.items():
                p = f()[m]; pred = (p >= .5).astype(int); yy = y[m]
                kk = int((pred == yy).sum())
                by[kind][v][k] = {"acc": kk / m.sum(), "acc_ci95": wilson(kk, int(m.sum())),
                                  "auc": core(p, yy)["auc"], "mean_conf": float(np.where(p >= .5, p, 1 - p).mean())}
    report["by_source_and_resolution"] = by

    # ---- 3. robustness suite (temperature as shipped / fitted on calib; never refit here)
    rob = {}
    for cond in perturb.ALL:
        rob[cond] = {}
        fam = "seen" if cond in perturb.SEEN else ("harsher" if cond in perturb.HARSHER else "new_family")
        for k, (n, t) in spec.items():
            p = P(n, cond, t)
            pred = (p >= .5).astype(int)
            d = core(p, y)
            d.update(family=fam, acc_ci95=wilson(int((pred == y).sum()), len(y)),
                     real_recall=float((pred[y == 1] == 1).mean()),
                     label_flip_vs_clean=float((pred != (P(n, "clean", t) >= .5)).mean()))
            rob[cond][k] = d
    report["robustness"] = rob
    json.dump(report, open(os.path.join("artifacts", "eval_report.json"), "w"), indent=1)

    # ---------------- printing
    def pc(x):
        return "  n/a " if x != x else f"{x * 100:5.1f}%"
    keys = list(rows)
    print("\n=== HEADLINE (clean test, full image) ===")
    for k in keys:
        d = h[k]; c = d["ci95"]
        print(f"{k:28s} acc={pc(d['acc'])} [{pc(c['acc'][0])},{pc(c['acc'][1])}] fakeP={pc(d['fake_precision'])} fakeR={pc(d['fake_recall'])} "
              f"[{pc(c['fake_recall'][0])},{pc(c['fake_recall'][1])}] fakeF1={d['fake_f1']:.4f} AUC={d['auc']:.4f} Brier={d['brier']:.4f} ECE={d['ece']:.4f} "
              f"CM={d['confusion']} acc@>=90%conf={pc(d['acc_when_conf_ge90'])} ({d['frac_conf_ge90'] * 100:.0f}% of preds)")
    print("\n=== HEADLINE (each model's own validated crop policy) ===")
    for k in pol:
        d = hp[k]
        print(f"{k:28s} acc={pc(d['acc'])} fakeR={pc(d['fake_recall'])} fakeP={pc(d['fake_precision'])} AUC={d['auc']:.4f} Brier={d['brier']:.4f} ECE={d['ece']:.4f}")
    print("\n=== PAIRED DIFFERENCE vs original (full image; bootstrap 95% CI) ===")
    for k in keys:
        if "paired_diff_vs_original" in h[k]:
            for m, v in h[k]["paired_diff_vs_original"].items():
                print(f"{k:28s} {m:15s} diff={v['mean']:+.4f} CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] P(better)={v['p_better']:.2f}")
    for kind in ("source", "resolution"):
        print(f"\n=== BY {kind.upper()} (clean, full image): accuracy [Wilson CI] / mean confidence ===")
        for v, d in by[kind].items():
            print(f"{v:34s} n={d['n']:4d} {d['classes']}")
            for k in keys:
                e = d[k]
                print(f"    {k:28s} acc={pc(e['acc'])} [{pc(e['acc_ci95'][0])},{pc(e['acc_ci95'][1])}] conf={pc(e['mean_conf'])}")
    print("\n=== ROBUSTNESS (acc / fake-recall / real-recall / AUC / ECE / label-flip-vs-clean) ===")
    for cond, d in rob.items():
        print(f"-- {cond} [{d[keys[0]]['family']}]")
        for k in keys:
            e = d[k]
            print(f"    {k:28s} acc={pc(e['acc'])} fakeR={pc(e['fake_recall'])} realR={pc(e['real_recall'])} AUC={e['auc']:.3f} ECE={e['ece']:.3f} flip={pc(e['label_flip_vs_clean'])}")


if __name__ == "__main__":
    main()
