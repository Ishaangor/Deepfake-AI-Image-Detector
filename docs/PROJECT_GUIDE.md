# Project guide

A plain-language tour of the project: what it is for, how it got here, the main decisions, and how to read what it
says. For commands see [INSTALLATION_AND_USAGE.md](INSTALLATION_AND_USAGE.md); for the code see
[ARCHITECTURE.md](ARCHITECTURE.md); for every number see [MODEL_AND_EVALUATION.md](MODEL_AND_EVALUATION.md).

## 1. The goal in one paragraph

Given one image, say whether it was **made by an AI image generator** or **taken with a camera**, and say how sure the
answer is. The tool must be honest: it gives a calibrated probability, answers *inconclusive* when the evidence is weak,
shows which parts of the image the model looked at, and publishes where it fails. It is a research and teaching
prototype, not a forensic instrument.

## 2. What a user sees

1. Upload an image (or pick one of eight demo images) in the web app, or run `predict.py` on the command line.
2. The app returns one of three verdicts - *Likely AI-generated*, *Inconclusive* or *Likely real (camera)* - with the
   calibrated probability that the image is AI (P(AI)) on a gauge.
3. Six tabs add detail: **Explanation** (a Grad-CAM heat map), **Analysis breakdown** (each tile's own score),
   **File diagnostics** (size, format, EXIF), **Reliability** (measured accuracy and known weak spots),
   **Method & limits** and **Technical**.

Everything runs locally; the image never leaves the computer.

## 3. How the detector works (no maths)

* **Tiles, not thumbnails.** Generators leave faint traces in fine pixel patterns. Shrinking the image to a fixed size
  would wash them out, so the model reads up to nine 256 x 256 pixel squares (*tiles*) at the original resolution.
* **One small network.** Each tile goes through EfficientNet-B0, a compact image network pre-trained on ImageNet and then
  trained here to separate AI images from real photos. Each tile is also scored mirrored, and the scores are averaged.
* **Calibration.** Neural networks tend to be over-confident. One number (the *temperature*, 1.618) was fitted on
  images set aside for this purpose so that "80 %" means roughly "right 80 % of the time" on those images.
* **Abstaining.** If P(AI) falls between 19 % and 73 %, the app says *inconclusive* instead of guessing. The edges were
  chosen on the same set-aside images.
* **Explanation.** Grad-CAM marks which regions pushed the score toward the answer. It shows what the *model* used, not
  where an image was edited.

## 4. How the project got here

| stage | what happened | lesson |
|---|---|---|
| 1. Face models | The project began with face-only detectors (Xception, then `best_finetuned.pt`) trained on teammate face datasets. | They only know face crops; modern AI images are mostly whole scenes. They are kept for comparison and rollback. |
| 2. `general_native_a` | First general model, trained on Community Forensics AI images and Open Images photos. Validation looked excellent (0.923). | On a harder independent check it scored only 0.688 AUC: the validation set was too similar to training. Good validation numbers are not enough. |
| 3. `general_native_b` | Retrained with more real web photos (LAION, ImageNet) and commercial-generator images, a harder validation set, and a fresh benchmark that nobody had looked at. | Benchmark AUC 0.909. This is the shipped model. |

Two details of stage 3 matter for honesty:
* The first benchmark (CompEval-v1) had been looked at before the decision to retrain, so it was **retired** and a new
  one (CompEval-v2) was built. Results are reported only on v2.
* Two training runs accidentally wrote to the same files. The shipped model is the **epoch-4 checkpoint of the second
  run**, which crashed in epoch 5. A better epoch of the first run (0.866 on validation) exists only in console output and
  is not claimed. Details: [EXPERIMENTS.md](EXPERIMENTS.md).

## 5. Key decisions and why

| decision | reason |
|---|---|
| Keep data in four separate roles: **train**, **val** (pick the epoch and explanation layer), **calib** (temperature and band), **test** (scored once) | so no number is tuned on the images it is reported on |
| Hold out whole **generator families** for the test (Midjourney, FLUX, Imagen, Stable Cascade, Kandinsky 2.2), chosen by a fixed rule in code | to measure generalisation to generators the model never saw, not memory |
| Re-compress some training PNGs to JPEG | AI images are often PNG and photos JPEG; without this the model could learn "PNG means AI" |
| Report ROC-AUC as the headline, plus recall, specificity and calibration | AUC does not depend on the threshold or the class mix; the others show what a user actually experiences |
| Report the clean-image result, not the higher JPEG results | the clean figure is the conservative one (section 6) |
| Show failures in the demo gallery | a gallery of only correct answers would misrepresent the detector |
| CPU support and a one-command setup | the demo and grading machines may have no GPU |

## 6. How to read the results

Headline on CompEval-v2 (project-constructed: 2,315 AI images from 8 held-out generators, 1,469 real photos):

| what it means | number |
|---|---|
| How well scores separate AI from real overall (1.0 = perfect, 0.5 = coin flip) | ROC-AUC **0.909** (95 % CI 0.900-0.918) |
| Share of AI images flagged at the 0.5 threshold | **72.5 %** |
| Share of real photos correctly left alone | **88.8 %** |
| How far confidence in the decided answer is from its accuracy | top-label ECE **0.022** |
| With the inconclusive band: share of images left undecided / accuracy of decided answers | 45.4 % / 93.3 % |

Read these with the caveats:
* **It is uneven.** Stable Cascade and Kandinsky are caught 95-98 % of the time; Imagen 3 and Midjourney only about
  half the time. About 1 in 5 ImageNet-style real photos is flagged as AI.
* **Mid-range scores lean low.** On unseen generators, an image scored 30-40 % AI was in fact AI 59 % of the time. Treat
  the middle as "don't know", never as "probably real".
* **Small images are hard.** At a quarter of the original size the AUC drops to 0.763.
* **Format confound.** All benchmark AI images are PNG files; re-compressing everything to JPEG *raises* AUC to 0.954. That
  higher figure is not the headline.
* **Outside the benchmark nothing is measured**: new generators, phone processing, screenshots or edited images may
  behave differently.

The full list is in the README, section 8.

## 7. Where things live

| folder | contents |
|---|---|
| `app.py`, `predict.py` | web app and command-line tool |
| `detector/` | inference, tiling, calibration band, Grad-CAM, file checks |
| `models/` | weights, model cards (`*.card.json`), checksums |
| `training/`, `evaluation/` | the scripts that built the data, trained, calibrated and benchmarked |
| `artifacts/` | the measured results every number in the docs comes from |
| `samples/` | eight demo images with labels and licences |
| `tests/` | automated tests (26) |
| `docs/` | this guide and the other documents |
| `legacy/` | the original face-model scripts, kept for reference |

## 8. Using the result responsibly

* Never use a single verdict to accuse anyone of faking an image. The detector misses AI images and raises false alarms
  at measured rates.
* Prefer the original file: screenshots, resizing and heavy compression weaken the signal.
* The shipped model was trained on CC BY-NC-SA 4.0 data, so treat it as non-commercial research material
  ([LICENSES.md](LICENSES.md)).

## 9. What would improve it next

1. A clean retrain with separate output paths, and more Imagen and Midjourney training images.
2. A benchmark with matched file formats.
3. A stronger backbone or per-generator calibration, measured on the same held-out families.
4. Upgrading PyTorch past the open security advisories and re-checking every number ([SBOM.md](SBOM.md)).
