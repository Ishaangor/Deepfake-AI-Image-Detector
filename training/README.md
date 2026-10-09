# training/

Data preparation and training for the general detector, plus the earlier face-only pipeline. Run every script **from the
repository root as a module** (`python -m training.<name>`). Paths come from `detector/paths.py`.

> Re-running this pipeline downloads tens of GB of third-party data (Community Forensics, CC BY-NC-SA 4.0; Open Images,
> CC BY 2.0) and takes several hours on a 6 GB GPU. **You do not need it to run the app**: the trained weights are in `models/`.

## General detector (shipped model `general_native_b`)
| step | command | output |
|---|---|---|
| 1. metadata scan | `python -m training.cf_scan small` and `python -m training.cf_scan eval` | `artifacts/cf_*_meta.parquet` (no image bytes) |
| 2. shard selection | `python -m training.cf_build select` | `artifacts/cf_selection.parquet` |
| 3. download | `python -m training.cf_download` | `data/commfor/raw/` |
| 4. build tile cache | `python -m training.cf_build build all` | `data/commfor/cache/s*_{native,glob}.npy` + meta |
| 5. web-style real photos | `python -m training.oi_build 6000` | cache shard 900 (Open Images validation photos) |
| 6. second pool + benchmark | `python -m training.ce_fetch`, then `python -m training.ce_build` | `data/commfor/ce/`, cache shard 910, `artifacts/ce_sel.parquet` |
| 7. train | `python -m training.train_general native_b --view native --epochs 6 --val_n 4000 --workers 2` | `models/general_native_b.pt`, `artifacts/train_general_native_b.json` |
| 8. calibrate | `python -m training.finalize_general general_native_b` | `models/general_native_b.meta.json` (temperature + band) |

Then evaluate with the scripts in `evaluation/` (see its README).

Helpers: `cf_data.py` (dataset over the cache), `parquet_stream.py` (memory-light parquet reading), `perturb.py`
(fixed degradation suite), `cf_eval_fetch.py` (CompEval-v1 fetch; that benchmark is retired).

## Earlier face-only phase
`face_splits.py` (leak-aware splits), `face_finetune.py`, `face_robust.py`, `face_finalize.py` (temperature and crop
policy). These produced `models/best_finetuned.pt` and `models/candidate_*.pt` from the teammate face datasets in `data/`.
