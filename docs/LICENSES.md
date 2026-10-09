# Licence audit

What each part of this repository is licensed under, as far as can be established from the repository and the upstream
sources. **This is an engineering audit, not legal advice.** Items marked *unknown* need a decision by the repository
owners.

## Summary

| part | licence | status |
|---|---|---|
| Project source code | **none declared** (no `LICENSE` file) | owners must choose |
| Python dependencies (73 packages) | permissive / weak copyleft (BSD, MIT, Apache-2.0, PSF, HPND, MPL-2.0) | OK |
| Shipped model `general_native_b.pt` | trained on CC BY-NC-SA 4.0 data | treat as **non-commercial, share-alike** |
| Face models (`best_finetuned.pt`, `candidate_*.pt`, `best_xception.keras`) | trained on teammate face datasets | **unknown** |
| Demo images (`samples/`) | CC BY 2.0 (4 photos), CC BY-NC-SA 4.0 (4 AI images) | OK with attribution |
| Test images (`tests/assets/`) | not recorded | **unknown** |
| Screenshots (`docs/screenshots/`) | contain the demo images above | attribution given in README |
| Community Forensics / CompEval data (not in git) | CC BY-NC-SA 4.0 | non-commercial research |
| Open Images V7 (not in git) | images CC BY 2.0 (per image), annotations CC BY 4.0 | OK with attribution |
| Teammate face datasets in `data/` (in git, ~700 MB) | **not documented** | confirm or remove before going public |

## 1. Project code

There is no `LICENSE` file, so by default all rights are reserved by the authors and nobody else may reuse the code.
An earlier README showed an "MIT" badge, but no licence text was ever committed, so that badge was not supported and
has been removed. No licence has been added on the team's behalf.

**Recommendation:** the owners should add a licence. A permissive licence (MIT or Apache-2.0) fits the code and all
dependencies. Whatever is chosen for the code, the trained weights and data remain bound by the data licences below.

## 2. Python dependencies

From [`sbom/sbom.cdx.json`](../sbom/sbom.cdx.json) (licences as declared in each package's metadata):

| licence | packages |
|---|---|
| BSD (3-clause, or 2-clause for Pygments) | torch, torchvision\*, numpy (with 0BSD, MIT, Zlib, CC0-1.0 parts), scipy, scikit-learn, pandas, altair, Jinja2, MarkupSafe, click, contourpy, cycler, fsspec, gitdb, GitPython, httpcore, httpx, idna, joblib, kiwisolver, mpmath, networkx, smmap, sympy, threadpoolctl, cloudpickle, colorama, Pygments, protobuf\* |
| Apache-2.0 | streamlit, timm, huggingface_hub, hf-xet, opencv-python-headless, pyarrow, requests, safetensors, tenacity, tornado, tzdata, watchdog, python-dateutil (dual with BSD), packaging (or BSD-2-Clause), pydeck\* |
| MIT | anyio, attrs, blinker, cachetools, charset-normalizer, filelock, fonttools, h11, iniconfig, jsonschema, jsonschema-specifications, narwhals, pip, pluggy, pyparsing, pytest, pytz, PyYAML, referencing, rpds-py, setuptools, six, toml, urllib3 |
| MIT-CMU (HPND) | pillow |
| PSF / Python-2.0 | matplotlib, typing_extensions |
| MPL-2.0 | certifi; tqdm (MPL-2.0 AND MIT) |

\* No licence in the package metadata; the upstream projects state BSD-3-Clause (torchvision, protobuf) and Apache-2.0
(pydeck).

No GPL or AGPL component is present at runtime. MPL-2.0 is file-level copyleft: it only matters if those packages' own
files are modified and redistributed. The optional legacy path uses TensorFlow (Apache-2.0), which is not installed by
default.

## 3. Model weights

| file | how it was made | licence position |
|---|---|---|
| `models/general_native_b.pt` (shipped) | EfficientNet-B0 initialised from timm's ImageNet weights, fine-tuned on Community Forensics, CompEval-derived images and real photos (Open Images, LAION, ImageNet, COCO, FFHQ, LHQ, others as sampled by Community Forensics) | The training data are CC BY-NC-SA 4.0. Whether model weights are a "derivative" of training data is legally unsettled; the cautious position, taken here, is to release and use them **only for non-commercial research, with attribution, under the same terms**. |
| timm `efficientnet_b0` ImageNet weights (starting point, not shipped separately) | from the timm library | timm is Apache-2.0. The weights were trained on ImageNet, whose terms allow non-commercial research use. |
| `best_finetuned.pt`, `candidate_A_mild.pt`, `candidate_B_full.pt` | fine-tuned on the teammate face datasets | **unknown** - inherits whatever the face datasets allow |
| `best_xception.keras` | original Xception model from the first project phase | **unknown** - training data not documented |

## 4. Images in the repository

**Demo images (`samples/`)** - full details per image in [`samples/manifest.json`](../samples/manifest.json):
* real photos from Open Images V7, Flickr, **CC BY 2.0**: "Cutting back to one cup of coffee today ;-)" by Daniel Farrell,
  "Fish Dish" by Yosomono, "Sunbound" by Kyle Garrity, "Tasman Peninsula" by Prince Roy. Unmodified.
* AI-generated images from the Community Forensics evaluation set, **CC BY-NC-SA 4.0**. The terms of the generator
  services that originally produced them may also apply.

**Screenshots (`docs/screenshots/`)** show the app with these demo images; the README credits them under the
screenshots.

**Test images (`tests/assets/`)** - `Fakers.jpg`, `face_*.jpg`, `fake_20.png`, `real_20.png` came from the earlier
`test_images/` folder. Their origin and licence are not recorded (and the labels of three of them are unknown). They are
used only by the automated tests. **Owners should confirm their origin or replace them** before a public release.

## 5. Datasets

| dataset | where | licence | notes |
|---|---|---|---|
| Community Forensics (training) and its evaluation set (benchmark, CompEval-derived pool) | `data/commfor/` (git-ignored, re-downloadable) | CC BY-NC-SA 4.0 | non-commercial research; attribution to Park & Owens (CVPR 2025) |
| Open Images V7 validation | `data/openimages/` (git-ignored) | images CC BY 2.0 (as listed per image), annotations CC BY 4.0 | attribution per image |
| `ai_faces/`, `deepfake_dataset/`, `rvf10k/`, `train.csv`, `valid.csv` | `data/`, **committed** in `cf7c601` | **not documented** | appear to be public Kaggle/GitHub face datasets, which often have non-commercial or unclear terms and contain photos of real people. Used only by the face-only scripts. **Confirm the licences or remove them (including from history) before making the repository public.** |

Datasets are otherwise not redistributed; the scripts in `training/` download them from their original sources.

## 6. What this means in practice

* Running the app for study, research, teaching or a demo: fine.
* Commercial use of the shipped model or a product built on it: **not permitted** by the training-data licence.
* Publishing the repository: add a code licence, keep the attributions, and resolve the face datasets and test images
  first.
