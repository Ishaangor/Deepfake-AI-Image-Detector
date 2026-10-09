"""Honest metrics for a general checkpoint on the independent CompEval subset.

    python eval_metrics.py general_native_a [v1|v2]    # reads artifacts/evalraw_<name>.parquet, models/<name>.meta.json

Positive class for precision/recall/F1 is FAKE (AI-generated). Decisions use calibrated P(real) < 0.5 -> "fake".
Reported: ROC-AUC, balanced accuracy, fake precision/recall/F1, confusion matrix, Brier, ECE (10 bins), 95% bootstrap CIs
(images resampled), per-generator recall, per-real-source specificity, per-condition table. The subset is sampled from
CompEval (see cf_eval_fetch.py), so per-generator intervals are wide; nothing here was used for model selection.
Writes artifacts/eval_<name>.json and eval_<name>.md.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.RandomState(0)


def ece(p_real, y_real, bins=10):
    conf = np.maximum(p_real, 1 - p_real)
    acc = ((p_real >= 0.5).astype(int) == y_real).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0.5 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def point(p_real, y_real):
    pred_real = p_real >= 0.5
    tp = int(((~pred_real) & (y_real == 0)).sum())            # fake caught
    fn = int((pred_real & (y_real == 0)).sum())
    fp = int(((~pred_real) & (y_real == 1)).sum())
    tn = int((pred_real & (y_real == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(n=int(len(y_real)), auc=float(roc_auc_score(y_real, p_real)) if len(set(y_real)) == 2 else None,
                balanced_acc=0.5 * (rec + spec), acc=(tp + tn) / len(y_real), fake_precision=prec, fake_recall=rec,
                fake_f1=2 * prec * rec / max(prec + rec, 1e-9), real_specificity=spec,
                brier=float(np.mean((p_real - y_real) ** 2)), ece=ece(p_real, y_real),
                confusion=dict(tp_fake=tp, fn_fake=fn, fp_fake=fp, tn_real=tn))


def boot(p_real, y_real, keys=("auc", "balanced_acc", "fake_recall", "real_specificity", "fake_f1"), B=300):
    idx0, idx1 = np.where(y_real == 0)[0], np.where(y_real == 1)[0]
    vals = {k: [] for k in keys}
    for _ in range(B):
        i = np.concatenate([RNG.choice(idx0, len(idx0)), RNG.choice(idx1, len(idx1))])
        r = point(p_real[i], y_real[i])
        for k in keys:
            vals[k].append(r[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def reliability(p_fake, y_fake, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0), min_n=1):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_fake >= lo) & ((p_fake <= hi) if hi == 1.0 else (p_fake < hi))
        if m.sum() >= min_n:
            out.append(dict(lo=lo, hi=hi, n=int(m.sum()), fake_rate=float(y_fake[m].mean())))
    return out


def main(name, bench="v1"):
    tag = "" if bench == "v1" else "__v2"
    raw = pd.read_parquet(os.path.join("artifacts", f"evalraw_{name}{tag}.parquet"))
    sel = pd.read_parquet(os.path.join("artifacts", "cf_eval_sel.parquet" if bench == "v1" else "ce_sel.parquet"))
    if "family" not in sel:
        sel["family"] = sel["model_name"]
    T = json.load(open(os.path.join("models", name + ".meta.json")))["temperature"]
    d = raw.dropna(subset=["logit"]).merge(sel[["file", "label", "key", "real_source", "format", "model_name", "family"]], on="file")
    d["y_real"] = 1 - d["label"].astype(int)
    d["p_real"] = 1 / (1 + np.exp(-d["logit"] / T))
    res = dict(name=name, bench=bench, temperature=T, n_images=int(d.file.nunique()), conditions={}, per_generator={},
               per_family={}, per_real_source={})
    for c, g in d.groupby("cond"):
        r = point(g.p_real.values, g.y_real.values)
        r["ci95"] = boot(g.p_real.values, g.y_real.values) if c == "clean" and g.y_real.nunique() == 2 else None
        res["conditions"][c] = r
    cl = d[d.cond == "clean"]
    for k, g in cl[cl.y_real == 0].groupby("model_name"):
        res["per_generator"][k] = dict(n=int(len(g)), fake_recall=float((g.p_real < 0.5).mean()), mean_p_real=float(g.p_real.mean()))
    for k, g in cl[cl.y_real == 0].groupby("family"):
        res["per_family"][k] = dict(n=int(len(g)), recall=float((g.p_real < 0.5).mean()))
    res["reliability_bins"] = reliability(1 - cl.p_real.values, 1 - cl.y_real.values)
    for k, g in cl[cl.y_real == 1].groupby("real_source"):
        res["per_real_source"][k] = dict(n=int(len(g)), specificity=float((g.p_real >= 0.5).mean()), mean_p_real=float(g.p_real.mean()))
    cl = cl.assign(side=pd.cut(cl[["w", "h"]].min(axis=1), [0, 300, 600, 1100, 1e9], labels=["<300", "300-600", "600-1100", ">1100"]))

    def rate(g):
        f, r = g[g.y_real == 0], g[g.y_real == 1]
        return dict(n=int(len(g)), fake_recall=float((f.p_real < 0.5).mean()) if len(f) else None,
                    real_specificity=float((r.p_real >= 0.5).mean()) if len(r) else None)
    res["by_min_side"] = {str(k): rate(g) for k, g in cl.groupby("side", observed=True)}
    res["by_format"] = {str(k): rate(g) for k, g in cl.groupby("format")}
    json.dump(res, open(os.path.join("artifacts", f"eval_{name}{tag}.json"), "w"), indent=1)
    L = [f"# {name} on CompEval-{bench} subset (T={T:.3f}, {res['n_images']} images)", "",
         "| condition | AUC | bal.acc | fake P | fake R | fake F1 | real spec. | Brier | ECE |", "|---|---|---|---|---|---|---|---|---|"]
    for c, r in res["conditions"].items():
        L.append(f"| {c} | {r['auc'] if r['auc'] is not None else float('nan'):.3f} | {r['balanced_acc']:.3f} | {r['fake_precision']:.3f} | {r['fake_recall']:.3f} | "
                 f"{r['fake_f1']:.3f} | {r['real_specificity']:.3f} | {r['brier']:.3f} | {r['ece']:.3f} |")
    ci = res["conditions"]["clean"]["ci95"] or {}
    L += ["", "clean 95% bootstrap CI: " + ", ".join(f"{k} [{v[0]:.3f}, {v[1]:.3f}]" for k, v in ci.items()), "",
          "## Per generator (fake recall, clean)", "", "| generator | n | recall | mean P(real) |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['fake_recall']:.2f} | {v['mean_p_real']:.2f} |"
          for k, v in sorted(res["per_generator"].items(), key=lambda kv: kv[1]["fake_recall"])]
    L += ["", "## Per real source (specificity, clean)", "", "| source | n | specificity | mean P(real) |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['specificity']:.2f} | {v['mean_p_real']:.2f} |" for k, v in res["per_real_source"].items()]
    open(os.path.join("artifacts", f"eval_{name}{tag}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v1")
