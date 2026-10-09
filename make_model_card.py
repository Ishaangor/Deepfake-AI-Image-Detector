"""Build artifacts/model_card_<model>.json (read by app.py) from the measured benchmark results. Nothing is hand-entered.

    python make_model_card.py general_native_b [v2]

Needs artifacts/eval_<name>[__v2].json (eval_metrics.py) and models/<name>.meta.json (finalize_general.py).
Optional: artifacts/explain_eval_<name>.json (explain_eval.py).
Weak spots (families caught < 50 %, real sources kept real < 80 %) are derived from the numbers and listed as limitations.
"""
import json
import os
import sys

FIXED_LIMITS = [
    "The score is a statistical estimate from pixel statistics, not proof of how an image was made.",
    "Only generator families and real-image sources listed above were measured; anything else is untested.",
    "Benchmark images are samples; per-family rates rest on small counts and carry wide uncertainty.",
    "Heavy compression, very low resolution (< 256 px) and screenshots/re-uploads shift the score; see the robustness table.",
    "Attention maps show where the model looked, not where an image was manipulated.",
    "Training data (Community Forensics) is CC BY-NC-SA 4.0: research/non-commercial use only.",
]


def main(name, bench="v2"):
    tag = "" if bench == "v1" else "__v2"
    ev = json.load(open(os.path.join("artifacts", f"eval_{name}{tag}.json")))
    meta = json.load(open(os.path.join("models", name + ".meta.json")))
    clean = ev["conditions"]["clean"]
    ci = (clean.get("ci95") or {}).get("auc", [clean["auc"], clean["auc"]])
    n_fake = clean["confusion"]["tp_fake"] + clean["confusion"]["fn_fake"]
    n_real = clean["confusion"]["fp_fake"] + clean["confusion"]["tn_real"]
    per_family = {k: dict(n=v["n"], recall=v["recall"]) for k, v in ev["per_family"].items()}
    per_real = {k: dict(n=v["n"], specificity=v["specificity"]) for k, v in ev["per_real_source"].items()}
    card = dict(
        model=name, bench=bench, temperature=meta["temperature"], band=meta.get("band"),
        benchmark=dict(name="CompEval-v2 (held-out)" if bench == "v2" else "CompEval-v1 (zero-shot)", n_fake=n_fake,
                       n_real=n_real, auc=clean["auc"], auc_ci=ci, fake_recall=clean["fake_recall"],
                       real_specificity=clean["real_specificity"], ece=clean["ece"], brier=clean["brier"],
                       balanced_acc=clean["balanced_acc"], fake_precision=clean["fake_precision"],
                       fake_f1=clean["fake_f1"], confusion=clean["confusion"]),
        benchmark_note=("Whole generator families (and disjoint real-image shards) were withheld from training, "
                        "calibration and model selection." if bench == "v2" else
                        "Retired sample, seen before the training-data decision; reported as a zero-shot result only."),
        per_family=per_family, per_real_source=per_real,
        robustness={c: dict(auc=r["auc"], balanced_acc=r["balanced_acc"]) for c, r in ev["conditions"].items()
                    if r["auc"] is not None},
        reliability_bins=ev.get("reliability_bins", []),
    )
    ex = os.path.join("artifacts", f"explain_eval_{name}.json")
    if os.path.exists(ex):
        e = json.load(open(ex))
        layer = meta.get("explain_layer")
        row = e.get("layers", {}).get(layer, {}).get("correct")
        if isinstance(row, dict):
            card["explain"] = dict(layer=layer, n=row["n"], beats_random_frac=row["beats_random_frac"])
    lim = list(FIXED_LIMITS)
    weak = sorted(k for k, v in per_family.items() if v["recall"] < 0.5)
    if weak:
        lim.insert(0, "Detection rate below 50% for: " + ", ".join(weak) + ".")
    bad = sorted(k for k, v in per_real.items() if v["specificity"] < 0.8)
    if bad:
        lim.insert(0, "Real photos wrongly flagged more than 20% of the time for source: " + ", ".join(bad) + ".")
    card["limitations"] = lim
    out = os.path.join("artifacts", f"model_card_{name}.json")
    json.dump(card, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v2")
