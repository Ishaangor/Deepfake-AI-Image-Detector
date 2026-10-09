# models/

Runtime model files. They are tracked in git (each `.pt` is about 16 MB), so a normal clone is ready to run.

| file | what it is | used by default? |
|---|---|---|
| `general_native_b.pt` | **Shipped detector.** EfficientNet-B0 with one output (log-odds that the image is *real*), trained on 256 px native-resolution tiles. Epoch 4 of a 6-epoch run; see `docs/MODEL_AND_EVALUATION.md`. | yes |
| `general_native_b.meta.json` | Temperature `T = 1.618`, inconclusive band on P(real) `[0.270, 0.810]`, Grad-CAM layer `blocks.4`, calibration-set statistics. Written by `training/finalize_general.py`. | yes |
| `general_native_b.card.json` | Measured benchmark summary shown in the UI's *Reliability* tab. Written by `evaluation/model_card.py` from `artifacts/eval_general_native_b__v2.json`; nothing is hand-entered. | yes |
| `active_model.json` | `{"model": "general_native_b.pt"}` - which checkpoint the app and CLI load. | - |
| `best_finetuned.pt` (+ `.meta.json`) | Original **face-only** detector from the earlier project phase. Kept untouched for rollback. | no |
| `candidate_A_mild.pt`, `candidate_B_full.pt` (+ `.meta.json`) | Face-only robustness candidates from the earlier phase (`training/face_robust.py`). Kept for comparison; the tests use `candidate_A_mild.pt` to exercise the face-crop path. | no |
| `calibration.json` | Temperature of the original face model (legacy format). | no |
| `best_xception.keras` | The very first Keras Xception model. Used only by the rollback path `DETECTOR_BACKEND=legacy` (needs TensorFlow). | no |
| `SHA256SUMS` | Checksums of the weight files; verified by `scripts/check_env.py`. | - |

`_archive/` (git-ignored, may not exist) holds superseded local development checkpoints. It is not needed.

## Switching models
* Web UI: sidebar -> *Detector model*.
* CLI: `python predict.py image.jpg --model candidate_A_mild.pt`
* Environment (PowerShell): `$env:DETECTOR_MODEL = "candidate_A_mild.pt"`
* Permanently: edit `active_model.json`.

## Rollback
`best_finetuned.pt` has never been modified. Set `active_model.json` to `{"model": "best_finetuned.pt"}` to return to the
original face model, or set `DETECTOR_BACKEND=legacy` for the original Keras pipeline.

## Licence note
The weights were trained on Community Forensics data (CC BY-NC-SA 4.0) and start from timm's ImageNet-pretrained
EfficientNet-B0 (Apache-2.0). Treat them as **non-commercial research use only**. See `docs/LICENSES.md`.
