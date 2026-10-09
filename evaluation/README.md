# evaluation/

Run these from the repository root as modules. All selection decisions were made on the `val`/`calib` splits, and the
benchmark was scored once per model.

| command | what it does | output |
|---|---|---|
| `python -m evaluation.benchmark general_native_b --bench v2` | Scores every CompEval-v2 image under 8 conditions (clean, JPEG 75/50, half, quarter, blur, gray, social) | `artifacts/evalraw_general_native_b__v2.parquet` |
| `python -m evaluation.metrics general_native_b v2` | AUC with bootstrap CI, confusion matrix at 0.5, per-generator / per-source rates, ECE, Brier, robustness | `artifacts/eval_general_native_b__v2.{json,md}` |
| `python -m evaluation.explain_faithfulness general_native_b` | Chooses the Grad-CAM layer on `val` and measures deletion/insertion faithfulness vs. random | `artifacts/explain_eval_general_native_b.json` |
| `python -m evaluation.model_card general_native_b v2` | Copies the measured numbers into the card the UI reads | `models/general_native_b.card.json` |
| `python -m evaluation.face_compare` | Earlier face-only phase: original vs. robustness candidates | `artifacts/eval_report.json` |

`benchmark.py` needs the downloaded benchmark images (`data/commfor/ce/`, git-ignored). The other scripts need only the
small artifacts.
