# artifacts/

Small, versioned records of how the shipped model was built and measured. Large or re-creatable files in this folder
(`*.parquet`, `*.npy`, `evalraw_*`, `hash_cache.json`) are git-ignored.

| file | written by | content |
|---|---|---|
| `train_general_native_b.json` | `training/train_general.py` | Arguments, augmentation, training-set size, per-epoch validation AUCs of the shipped model |
| `eval_general_native_b__v2.json` / `.md` | `evaluation/metrics.py` | **Headline benchmark (CompEval-v2)**: AUC with CI, confusion matrix, per-generator, per-source, calibration, robustness |
| `explain_eval_general_native_b.json` | `evaluation/explain_faithfulness.py` | Grad-CAM layer choice and deletion/insertion faithfulness (76 % beats random) |
| `train_general_native_a.json`, `eval_general_native_a.*`, `explain_eval_general_native_a.json` | same scripts | First general model (superseded): CompEval-v1 AUC 0.688 - kept as the honest record of an attempt that did not generalise |
| `train_general_glob_a.json` | `training/train_general.py` | Resized-image ("glob") variant; not shipped |
| `cf_manifest.json`, `cf_qpool.json`, `cf_selection_info.json` | `training/cf_download.py`, `training/cf_build.py` | Which Community Forensics shards were used and how the JPEG-format bias was controlled |
| `splits_v2.json`, `eval_report.json`, `train_A_mild.json`, `train_B_full.json` | `training/face_*`, `evaluation/face_compare.py` | Records of the earlier face-only phase |

Git-ignored but needed to *re-run* the evaluation: `ce_sel.parquet` (benchmark selection), `evalraw_*.parquet` (raw
per-image scores), `cf_*_meta.parquet`. They are rebuilt by the pipeline in `training/README.md`.
