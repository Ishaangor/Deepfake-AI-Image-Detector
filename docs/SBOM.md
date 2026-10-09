# Software bill of materials (SBOM) and vulnerability scan

## Files

| file | content |
|---|---|
| [`sbom/sbom.cdx.json`](../sbom/sbom.cdx.json) | CycloneDX **1.6** JSON SBOM: the application (`ai-image-detector` 1.0.0, from `pyproject.toml`) and **73 components** (every Python package installed in the runtime environment, with version, PURL and declared licence) |
| [`sbom/pip-audit.json`](../sbom/pip-audit.json) | pip-audit report for exactly those package versions |

## How it was produced

| item | value |
|---|---|
| Generator | `cyclonedx-py` (cyclonedx-bom 7.5.0, cyclonedx-python-lib 11.12.0), mode `environment`, `--output-reproducible` (no timestamp or serial number, so regenerating gives a stable diff) |
| Validation | passed strict JSON-schema validation for CycloneDX 1.6 (`cyclonedx.validation.json.JsonStrictValidator`) |
| Scanner | pip-audit 2.10.1, `--no-deps` against the frozen package list (local version tags such as `+cpu` removed for lookup), PyPI advisory database |
| Environment scanned | the clean setup-test environment: a new `.venv` created by `scripts/setup.ps1 -Cpu` on Windows 11 with Python 3.13, i.e. `requirements.txt` plus PyTorch 2.6.0 CPU |
| Date | 9 October 2026 |
| Tools isolation | the SBOM tools run from a separate virtual environment (`.sbom-tools`, git-ignored), so they do not appear in the SBOM |

Regenerate (needs internet):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_sbom.ps1                      # scans .venv
powershell -ExecutionPolicy Bypass -File scripts\generate_sbom.ps1 -TargetVenv C:\path\to\.venv
```

## Coverage and exclusions

* **Covered:** every Python distribution in the runtime environment, including test tooling (`pytest`) and build
  tooling (`pip`, `setuptools`).
* **GPU installs:** the CUDA build differs only in `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124`. On Windows the CUDA
  runtime libraries ship inside the torch wheel and are not listed as separate components.
* **Not covered:** the Python interpreter, the operating system, NVIDIA drivers, the optional legacy TensorFlow path
  (not installed by `requirements.txt`), and non-software assets - model weights, demo images and datasets. Their
  provenance and licences are in [LICENSES.md](LICENSES.md), `models/README.md` and `samples/manifest.json`.
* The dependency graph inside the SBOM is incomplete (the generator warns that the root component has no declared
  dependencies); the component list itself is complete for the scanned environment.

## Vulnerability scan results (reported as found)

**First scan** (with the originally pinned versions): 42 unique advisories in 3 packages - `pillow 11.3.0` (18),
`setuptools 78.1.0` (2), `torch 2.6.0` (22). pip-audit printed 63 rows because some advisories are listed more than once.

**Action taken:**
* `pillow` 11.3.0 -> **12.3.0** (the version that fixes all 18). Pillow decodes untrusted uploads, so this mattered most.
* `setuptools` 78.1.0 -> **83.0.0** (fixes both).
* Defence in depth: uploads are now decoded **only** as JPEG, PNG, WEBP, BMP or TIFF (`detector/forensics.py`,
  `open_upload`). Pillow otherwise detects the format from the file content, so a renamed PSD, PDF, font or other rarely
  used format could reach decoders that several of the advisories concern. A test enforces this.
* The test suite (26 tests) passed after the upgrade, in both the development and the clean environment.

**Final scan: 22 unique advisories remain, all in `torch 2.6.0`** (24 rows in `sbom/pip-audit.json`).

| group | advisories | relevance to this app |
|---|---|---|
| `torch.load(weights_only=True)` unpickler bypass (CVE-2026-24747, fixed in 2.10.0) | 1 | the app loads only its own checkpoints from `models/`, whose SHA-256 hashes are listed in `models/SHA256SUMS` and checked by `scripts/check_env.py`. **Do not load checkpoints from untrusted sources** with this version |
| `torch.compile` / Inductor / `torch.jit.script` issues (fixed in 2.7-2.13) | several | not used: the app runs the model eagerly |
| crashes or wrong results in specific operators (`ctc_loss`, `lstm_cell`, `rnn.pad_packed_sequence` / `unpack_sequence`, `nan_to_num`, `linalg.lu`, `PairwiseDistance`, `bitwise_right_shift`, `mkldnn_max_pool2d`, quantized `Sigmoid`, profiler, sparse conversion, `rot90`) | most | not used by EfficientNet-B0 inference |
| of the above, no fixed version listed: profiler (PYSEC-2025-189), quantized `Sigmoid` (PYSEC-2025-190), `rnn.pad_packed_sequence` (PYSEC-2025-192), `.pt2` model loading (PYSEC-2026-139) | 4 | not used (the app never loads `.pt2` files) |

**Assessment:** the remaining issues concern features this app does not call, or loading untrusted model files, which
it does not do. They are still an **open item**: upgrading PyTorch (to 2.10 or later) changes the CUDA build and would
require re-verifying every measured number, which was not done before the demo. Recommended as the first maintenance
task. Exact advisory IDs and fixed versions are in `sbom/pip-audit.json`.

## Licences found in the SBOM

All 73 components carry permissive or weak-copyleft licences (BSD, MIT, Apache-2.0, PSF, HPND/MIT-CMU, MPL-2.0 for
`certifi` and part of `tqdm`). Three packages declare no licence in their metadata: `protobuf` (BSD-3-Clause upstream),
`pydeck` (Apache-2.0 upstream) and `torchvision` (BSD-3-Clause upstream). No GPL-licensed runtime component was found.
Full audit: [LICENSES.md](LICENSES.md).
