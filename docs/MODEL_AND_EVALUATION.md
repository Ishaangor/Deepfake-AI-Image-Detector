# Model and evaluation

Every number in this document comes from a file in the repository:
* `artifacts/eval_general_native_b__v2.json` and `.md`, written by `evaluation/metrics.py`;
* `models/general_native_b.meta.json`, written by `training/finalize_general.py`;
* `artifacts/train_general_native_b.json`;
* `artifacts/explain_eval_general_native_b.json`.

Two rows of the per-class table are derived from the confusion matrix, as marked. Nothing else was entered by hand.

## 1. The model

| item | value |
|---|---|
| Name | `general_native_b` (`models/general_native_b.pt`, SHA-256 `e23e2e0d...`) |
| Architecture | EfficientNet-B0 (timm 1.0.30, ImageNet-pretrained start), 1 output logit = log-odds of *real* |
| Input | 256 x 256 RGB tiles taken at **native resolution** (no resizing) |
| Inference | up to 9 tiles (3 x 3 grid, positions aligned to 8 px), each with horizontal-flip test-time augmentation; mean logit / T |
| Temperature T | 1.618, fitted on the `calib` split (n = 4,827, 33 generators); calibration-set NLL 0.463 -> 0.428 |
| Decision band | calibrated P(real) < 0.270 -> *likely AI*; > 0.810 -> *likely real*; otherwise *inconclusive*. Set on `calib` for 90 % class-balanced precision of each decided side. |
| Explanation | Grad-CAM at layer `blocks.4`, chosen on `val` by faithfulness (section 6) |
| Size / speed | 16 MB of weights (4.0 M parameters). For a 1024 x 1024 image (9 tiles), after the model is loaded: about 0.05 s for the score and 0.4 s for Grad-CAM on an RTX 4050 laptop GPU; about 0.35 s and 1.9 s on the laptop CPU. Measured on 3 repeats on the development laptop only. |

### Why this design
* **Native-resolution tiles.** Generator traces live in fine pixel statistics, and resizing an image to a fixed input
  size destroys them. On the same data and `val` split, the resized-image variant (`general_glob_a`) reached a val
  composite of 0.870, against 0.923 for the native-tile variant (`general_native_a`). Tiling also lets one model handle
  any image size.
* **EfficientNet-B0.** Strong ImageNet features at 4 M parameters. It trains on a 6 GB laptop GPU in about 6 minutes per
  epoch on 47 k tiles and runs on a CPU. Larger backbones were not tried (time and GPU limits).
* **Temperature scaling and an abstain band.** Raw network outputs are over-confident. One temperature parameter fixes
  most of that without changing the ranking (AUC). The band lets the app say "inconclusive" instead of guessing.

## 2. Data and splits

| split | used for | content |
|---|---|---|
| `train` | weights | 47,716 tiles (22,801 real). AI images from many Community Forensics generators. Real images from COCO, Open Images, LAION, ImageNet, FFHQ, LHQ and others. |
| `val` | choosing the epoch, the Grad-CAM layer and the demo images | 4,000 tiles (2,000 real) from generators and cameras **not** in `train` |
| `calib` | temperature and decision band | 4,827 images from 33 generators, disjoint from `train`/`val` |
| **CompEval-v2 (benchmark)** | final measurement only | 3,784 images: 2,315 AI images from **held-out generator families** and 1,469 real photos from held-out row groups |

* **Held-out families.** Midjourney (v5.2, v6.1, and a community v6 model), FLUX (dev, schnell), Imagen 3, Stable Cascade
  and Kandinsky 2.2. They were chosen by a fixed hash rule in `training/ce_fetch.py` *before* any image was viewed.
  No image of these families was used for training, calibration or selection.
* **Format-bias controls.** AI images are often PNG while real photos are JPEG, and a model can learn that shortcut
  instead of real evidence. In training, PNG images were recompressed to JPEG with qualities drawn from the real-image
  pool (`artifacts/cf_selection_info.json`). Tiles are cropped at native resolution, and each generator's contribution
  is capped.
* **Training mix** (sampling shares from the training log):
  * AI images: Community Forensics "Systematic" 15 %, "Manual" 10 %, CompEval-derived 25 %;
  * real images: LAION 17.5 %, Open Images 10.5 %, ImageNet 5 %, COCO 5 %, FFHQ 4 %, LHQ 4 %, vision datasets 4 %.

**The benchmark is project-constructed.** It is a stratified sample that this project drew from the public
Community Forensics evaluation set. It is not an official leaderboard split.

## 3. Results on CompEval-v2 (clean images)

| metric | value | 95 % bootstrap CI |
|---|---|---|
| ROC-AUC | **0.909** | 0.900 - 0.918 |
| Balanced accuracy (threshold 0.5) | 0.807 | 0.794 - 0.819 |
| Accuracy (threshold 0.5) | 0.788 | - |
| AI images detected (recall) | **72.5 %** | 70.7 - 74.3 % |
| Real images kept real (specificity) | **88.8 %** | 87.2 - 90.3 % |
| AI precision | 91.1 % | - |
| AI F1 | 0.807 | 0.794 - 0.820 |
| Brier score | 0.141 | - |
| Top-label ECE (10 bins) | **0.022** | - |
| Class-wise ECE on P(AI) (10 bins) | 0.136 | - |

### Confusion matrix (threshold P(AI) = 0.5)

| | predicted AI | predicted real | total |
|---|---|---|---|
| **actually AI** | 1,678 | 637 | 2,315 |
| **actually real** | 164 | 1,305 | 1,469 |

### Per class
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| AI-generated | 0.911 | 0.725 | 0.807 | 2,315 |
| Real | 0.672 *(derived: 1305 / 1942)* | 0.888 | 0.765 *(derived)* | 1,469 |

### With the inconclusive band (what the app actually shows)
Of all benchmark images, **45.4 % are inconclusive**. Decided answers are correct **93.3 %** of the time: "likely AI"
calls 96.1 %, "likely real" calls 90.0 %.

| | called AI | inconclusive | called real |
|---|---|---|---|
| AI images | 45.8 % | 50.1 % | 4.1 % |
| Real photos | 2.9 % | 38.1 % | 59.0 % |

The app abstains often. When it does decide, it is usually right.

### Per generator (AI images detected at 0.5)
| generator | n | detected |
|---|---|---|
| Imagen 3 | 252 | **50 %** |
| Midjourney v6.1 | 249 | **53 %** |
| Midjourney v5.2 | 350 | **56 %** |
| FLUX-schnell | 302 | 73 % |
| Midjourney v6 (community model) | 257 | 74 % |
| FLUX-dev | 339 | 79 % |
| Kandinsky 2.2 | 255 | 95 % |
| Stable Cascade | 311 | 98 % |

### Per real-image source (kept real at 0.5)
| source | n | kept real |
|---|---|---|
| FFHQ (faces) | 126 | 100 % |
| RAISE (raw camera photos) | 126 | 97 % |
| COCO | 126 | 91 % |
| LAION (web images) | 867 | 88 % |
| ImageNet | 224 | **79 %** |

With a few hundred images per row, each rate is uncertain by roughly +/- 6 percentage points.

### Calibration
* T = 1.618 was fitted on `calib` and is never tuned on the benchmark.
* **Top-label ECE 0.022.** Confidence in the decided class matches accuracy closely.
* **Class-wise.** The reliability bins in the table below show that on these **unseen** generators the model's P(AI)
  runs too low in the middle range. The mean predicted P(AI) is 0.48, while 61 % of benchmark images are AI-generated.
  The class-wise ECE is 0.136; part of the gap comes from the benchmark's 61 % AI share.

  In practice, a score of 30-40 % AI should **not** be read as "probably real". On this benchmark, 59 % of such images
  were AI-generated.

| predicted P(AI) | 0-10 % | 10-20 % | 20-30 % | 30-40 % | 40-50 % | 50-60 % | 60-70 % | 70-80 % | 80-90 % | 90-100 % |
|---|---|---|---|---|---|---|---|---|---|---|
| actually AI | 2.7 % | 24.0 % | 41.4 % | 58.7 % | 70.4 % | 78.7 % | 85.8 % | 91.1 % | 96.7 % | 98.3 % |
| n | 630 | 367 | 353 | 264 | 328 | 315 | 317 | 326 | 480 | 404 |

### Robustness (the same benchmark images, degraded before scoring)
| condition | ROC-AUC | balanced acc. | AI detected | real kept real |
|---|---|---|---|---|
| original | 0.909 | 0.807 | 72.5 % | 88.8 % |
| JPEG quality 75 | 0.938 | 0.869 | 86.2 % | 87.5 % |
| JPEG quality 50 | 0.954 | 0.882 | 84.2 % | 92.3 % |
| social-media style (<= 1080 px + JPEG 70) | 0.941 | 0.867 | 86.0 % | 87.5 % |
| grayscale | 0.926 | 0.832 | 74.8 % | 91.5 % |
| Gaussian blur (sigma 1) | 0.881 | 0.793 | 76.3 % | 82.3 % |
| half size | 0.842 | 0.770 | 73.1 % | 80.9 % |
| **quarter size** | **0.763** | 0.711 | 74.2 % | **67.9 %** |

**Why is AUC *higher* after JPEG compression?** All 2,315 AI images in the benchmark are PNG files. The real photos
are mostly JPEG (1,154) with 174 PNG and a few TIFF/WEBP/GIF. Recompressing everything to JPEG removes a format
difference that the model apparently reads *against* the AI class. The original-image row is therefore the conservative
headline figure. **Downscaling hurts most**: at quarter size, a third of real photos are flagged.

## 4. Training history (accurate record)

| run | data | val composite* | independent result | status |
|---|---|---|---|---|
| face models (`best_finetuned`, `candidate_A/B`) | teammate face datasets | - | face-only | kept for rollback |
| `general_native_a` | Community Forensics Small + Open Images | 0.923 | **CompEval-v1 AUC 0.688** (0.667-0.709) | did not generalise; superseded |
| `general_glob_a` | same, resized ("glob") view | 0.870 (epoch 5 of 8) | not benchmarked | not shipped |
| **`general_native_b`** | + CompEval-derived pool | **0.850 (epoch 4)** | **CompEval-v2 AUC 0.909** (0.900-0.918) | **shipped** |

\* composite = 0.5 x clean AUC + 0.5 x mean AUC over {JPEG 50, downscale to 128, blur}, on `val`. `native_a` and
`glob_a` share one `val` set (2,400 tiles). `native_b` uses a larger and harder one (4,000 tiles, including unseen
web-image sources), so its 0.850 cannot be compared with 0.923. The two benchmarks also differ: `native_a` was measured on CompEval-v1,
where it kept only 38 % of LAION web photos as real, and `native_b` on CompEval-v2. The step from 0.688 to 0.909 is
therefore indicative, not a controlled comparison.

**What happened with `general_native_b`.** Two training runs of the same configuration were started on the same
checkpoint and log paths:
1. An 8-epoch run reached val composites 0.746, 0.818, 0.856, 0.841 and 0.866 (epochs 1-6, last one logged at epoch 6).
   Its log file and best checkpoint were later overwritten by the second run, so these values survive only in the
   session's console output and are **not** reproducible from the repository.
2. A 6-epoch run (2 data-loader workers). Its log in `artifacts/train_general_native_b.json` records val composites
   0.764, 0.814, 0.843 and **0.850** for epochs 1-4. During epoch 5 it crashed with a Windows paging-file error (error
   1455, out of virtual memory).

The shipped checkpoint is the **epoch-4 checkpoint of run 2** (val composite 0.85008825, recorded in
`models/general_native_b.meta.json`). It was calibrated and benchmarked exactly once. A clean, complete retrain was not
done before the demo; it might score slightly higher.

## 5. Model-selection hygiene
* Epoch, Grad-CAM layer and demo samples were chosen on `val`. Temperature and band were set on `calib`.
* CompEval-v1 was looked at before the decision to retrain. It was therefore **retired** as a test set and kept only as
  the zero-shot record of `general_native_a`. CompEval-v2 was built afterwards from different row groups.
* The benchmark was scored once for the shipped model. The class-wise ECE and verdict statistics were added later as
  *additional reporting* computed from the same stored scores; nothing was tuned on them.

## 6. Explanation faithfulness
Grad-CAM at three candidate layers was compared on `val` tiles: 80 the model classified correctly and 80 it got wrong.
Each map explains the predicted class. Pixels are removed (**deletion**) or revealed (**insertion**) step by step, in
the order the map ranks them and, separately, in random order. A map "beats random" on an image when removing its top
regions lowers the model's score faster than random removal **and** revealing them restores it faster.
`evaluation/explain_faithfulness.py` documents the method.

| layer | beats random (correct cases, n = 80) | beats random (wrong cases, n = 80) |
|---|---|---|
| **`blocks.4`** (used) | **76 %** | 69 % |
| `bn2` (last layer) | 58 % | 54 % |
| `blocks.2` | 46 % | 60 % |

A weight-randomisation sanity check was also run: maps from a network with randomised weights were essentially
uncorrelated with the real maps (mean Spearman rho of about 0.01).

This shows the map is *connected to the model's decision*. It does not show that the highlighted pixels are
manipulated. See `docs/FAQ_AND_VIVA.md`.

## 7. Reproducing
```
python -m evaluation.metrics general_native_b v2        # needs artifacts/evalraw_general_native_b__v2.parquet (git-ignored)
python -m evaluation.model_card general_native_b v2
```
Re-scoring from images (`python -m evaluation.benchmark general_native_b --bench v2`) needs the benchmark images,
which are downloaded by `python -m training.ce_fetch` together with the training pool. On the development machine
`data/commfor/ce/` holds about 15 GB. The data are licensed CC BY-NC-SA 4.0.
