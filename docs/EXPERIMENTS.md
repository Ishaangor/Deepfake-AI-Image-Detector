# Experiment record

All model selection used the `val` split; temperature and the decision band used `calib`; each benchmark was scored once per model.

| # | model | training data | selection (val composite) | independent result |
|---|---|---|---|---|
| 0 | original `best_finetuned.pt` / Xception | faces (rvf10k, AI faces) | - | face-only; not evaluated on general images |
| 1 | `general_native_a` (dev stand-in, not shipped) | Community Forensics Small + Open Images | 0.923 | **CompEval-v1 AUC 0.688** (CI 0.667-0.709): failed to generalise - LAION reals only 38 % kept real, many modern generators missed |
| 2 | `general_native_b` (**shipped**) | + CompEval-derived pool (LAION/ImageNet reals, commercial-generator fakes) | 0.850 (epoch 4; see note) | **CompEval-v2 AUC 0.909** (CI 0.900-0.918) |

Note on run 2: an 8-epoch run was interrupted at epoch 6-7 by a Windows paging-file error (error 1455) while a second, shorter run
was also writing the same checkpoint. The saved checkpoint is the best epoch of the shorter run (val composite 0.850); the longer run
had reached 0.866 but that checkpoint was overwritten. A clean retrain would likely improve results slightly.

## Decisions
* CompEval-v1 was seen before the retraining decision, so it was **retired** as a benchmark and is kept only as the zero-shot
  record of run 1. CompEval-v2 was created afterwards: held-out generator families (Midjourney, FLUX, Imagen, Stable
  Cascade, Kandinsky 2.2) and disjoint real row groups. (`kandinsky_2_2` landed in the benchmark by the hash rule fixed in code.)
* Format-bias controls: PNG images recompressed to JPEG with qualities drawn from the real-image pool; native-resolution
  crops; per-generator caps; generator-level and device-level splits.
* Explanation layer `blocks.4` chosen by deletion/insertion faithfulness against random removal: 76 % beats-random on native_b
  validation tiles vs 57 % (`bn2`) and 46 % (`blocks.2`).
* The model is weakest on Imagen 3 and Midjourney (about 50 %); this is reported, not hidden.
* Rollback: `models/best_finetuned.pt` is untouched; use `models/active_model.json`, `DETECTOR_MODEL`, or `DETECTOR_BACKEND=legacy`.
* Not done: a clean full-length retrain; other architectures; Midjourney/Imagen-specific training data.
