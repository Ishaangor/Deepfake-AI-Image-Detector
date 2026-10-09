# Experiment record

All model selection used the `val` split; temperature and the decision band used `calib`; each benchmark was scored once
per model. Numbers and their source files are in [MODEL_AND_EVALUATION.md](MODEL_AND_EVALUATION.md).

| # | model | training data | selection (val composite) | independent result |
|---|---|---|---|---|
| 0 | original `best_finetuned.pt` / Xception | faces (rvf10k, AI faces) | - | face-only; not evaluated on general images |
| 1 | `general_native_a` (not shipped) | Community Forensics Small + Open Images | 0.923 | **CompEval-v1 AUC 0.688** (CI 0.667-0.709): failed to generalise - only 38 % of LAION real photos kept real, many modern generators missed |
| 1b | `general_glob_a` (not shipped) | same data, whole image resized ("glob") | 0.870 (epoch 5 of 8) | not benchmarked; showed that resizing loses signal |
| 2 | `general_native_b` (**shipped**) | + CompEval-derived pool (LAION/ImageNet real photos, commercial-generator images) | 0.850 (epoch 4; see note) | **CompEval-v2 AUC 0.909** (CI 0.900-0.918) |

The val composite is 0.5 x clean AUC + 0.5 x mean AUC over {JPEG 50, downscale to 128, blur}. Runs 1 and 1b share a
`val` set of 2,400 tiles; run 2 uses a larger and harder one (4,000 tiles), so 0.850 and 0.923 are not comparable.

**Note on run 2.** Two runs of the same configuration wrote to the same checkpoint and log paths:
1. an 8-epoch run reached val composites 0.746, 0.818, 0.856, 0.841 and 0.866 (epochs 1-6, last logged at epoch 6). Its
   log and best checkpoint were overwritten by the second run; these values survive only in console output and cannot
   be reproduced from the repository;
2. a 6-epoch run (2 data-loader workers) logged 0.764, 0.814, 0.843 and 0.850 for epochs 1-4
   (`artifacts/train_general_native_b.json`) and crashed during epoch 5 with a Windows paging-file error (1455).

The shipped checkpoint is the epoch-4 checkpoint of the second run (val composite 0.850). A clean retrain was not done
and might score slightly higher.

## Decisions
* CompEval-v1 was seen before the retraining decision, so it was **retired** as a benchmark and is kept only as the
  zero-shot record of run 1. CompEval-v2 was created afterwards: held-out generator families (Midjourney, FLUX, Imagen,
  Stable Cascade, Kandinsky 2.2) and disjoint real row groups, chosen by a hash rule fixed in `training/ce_fetch.py`.
* Format-bias controls: PNG images recompressed to JPEG with qualities drawn from the real-image pool; native-resolution
  crops; per-generator caps; generator-level and device-level splits.
* Explanation layer `blocks.4` chosen by deletion/insertion faithfulness against random removal on `val` tiles: 76 %
  beats random, against 58 % for `bn2` and 46 % for `blocks.2` (`evaluation/explain_faithfulness.py`).
* Additional reporting added after the single benchmark run, computed from the stored scores without re-scoring or
  tuning: class-wise ECE and three-way verdict statistics (`evaluation/metrics.py`).
* The model is weakest on Imagen 3 and Midjourney (about 50 %); this is reported, not hidden.
* Rollback: `models/best_finetuned.pt` is untouched; use `models/active_model.json`, `DETECTOR_MODEL`, or
  `DETECTOR_BACKEND=legacy`.
* Not done: a clean full-length retrain; other architectures; Midjourney/Imagen-specific training data.

## Scripts (after the repository restructure)
Run from the repository root. Each script's docstring lists its full arguments and inputs.

| step | command |
|---|---|
| Community Forensics pool | `python -m training.cf_build select`, then `python -m training.cf_build build all` |
| Open Images real photos | `python -m training.oi_build` |
| CompEval-derived pool and benchmark images | `python -m training.ce_fetch`, then `python -m training.ce_build` |
| train | `python -m training.train_general general_native_b --view native` |
| calibrate, set band, pick Grad-CAM layer | `python -m training.finalize_general general_native_b` |
| score the benchmark | `python -m evaluation.benchmark general_native_b --bench v2` |
| metrics and model card | `python -m evaluation.metrics general_native_b v2`, then `python -m evaluation.model_card general_native_b v2` |
| explanation faithfulness | `python -m evaluation.explain_faithfulness general_native_b` |
