# Installation and usage (Windows 11)

Written for someone setting the project up on a Windows laptop for the first time. Every command is for **PowerShell**,
run from the repository folder.

## 1. What you need

| item | notes |
|---|---|
| Windows 10 or 11 | tested on Windows 11 |
| Python 3.11 or newer | **3.13 tested**. Install from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"** |
| Git | to clone the repository (or download the ZIP from GitHub) |
| ~3 GB free disk | PyTorch alone is about 2.5 GB with CUDA, about 0.3 GB for CPU |
| internet | only for the first setup |
| NVIDIA GPU | optional. Without one, everything runs on the CPU (about 0.35 s per 1024 px image after loading) |

## 2. Automatic setup (recommended)

```powershell
git clone https://github.com/Ishaangor/Deepfake-AI-Image-Detector.git C:\aidetector
cd C:\aidetector
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

A short path such as `C:\aidetector` avoids the Windows 260-character path limit (see troubleshooting).

`setup.ps1` does the following and stops with a `fix:` hint if a step fails:
1. finds Python 3.13, 3.12 or 3.11 (via the `py` launcher, then `python`);
2. creates the virtual environment `.venv` in the repository folder;
3. installs PyTorch 2.6.0 - the CUDA 12.4 build if `nvidia-smi` is found, otherwise the CPU build;
4. installs the pinned packages from `requirements.txt`;
5. runs `scripts\check_env.py --smoke`: Python version, every package, the model file's SHA-256 checksum, the calibration
   file (temperature and band), the model card, the compute device, port 8501, and one real prediction.

Options: `-Cpu` forces the CPU build of PyTorch (use this if the GPU build gives trouble); `-Python "py -3.12"` picks a
specific interpreter.

`-ExecutionPolicy Bypass` applies to that one command only; it does not change your system settings.

## 3. Start the app

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

Your browser opens <http://localhost:8501>. Stop the app with **Ctrl+C** in the PowerShell window.

| option | effect |
|---|---|
| `-Cpu` | force CPU inference (sets `DETECTOR_DEVICE=cpu`) |
| `-Port 8502` | use another port |

The app listens on `localhost` only, so other computers on the network cannot open it.

## 4. Use the command line

```powershell
.venv\Scripts\python predict.py samples\real_coffee_cup.jpg
.venv\Scripts\python predict.py C:\path\to\image.png --json
.venv\Scripts\python predict.py C:\path\to\image.png --model candidate_A_mild.pt --crop face
```

Text output shows the model, the number of regions scored, the calibrated AI probability, the spread across regions,
the verdict and the inconclusive band. `--json` prints one line with `fake_prob`,
`real_prob`, `verdict`, `band`, `temperature`, per-tile probabilities, `flags` and `model_id`.

## 5. Manual setup (any OS, or if the script cannot be used)

Windows:
```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# GPU (NVIDIA, CUDA 12.4 build):
.venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# or CPU only:
.venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\check_env.py --smoke
.venv\Scripts\python -m streamlit run app.py
```

macOS / Linux (**not tested**; the same pinned versions should work):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 && pip install -r requirements.txt
python scripts/check_env.py --smoke && streamlit run app.py
```

## 6. Verify an installation

```powershell
.venv\Scripts\python scripts\check_env.py --smoke     # every line should read [OK  ]
.venv\Scripts\python -m pytest                        # 26 tests
```

**What was tested for this release:**
* a fresh copy of the repository in `%TEMP%\aidc`, new `.venv`, `setup.ps1 -Cpu` (CPU PyTorch 2.6.0+cpu): all
  environment checks passed; the smoke prediction for the sheep demo image was 0.8581 P(AI) (0.8585 on the GPU);
  26 tests passed after the final dependency upgrade (Pillow 12.3.0, setuptools 83.0.0);
* the development laptop (Windows 11, Python 3.13, RTX 4050 6 GB, CUDA build): all checks passed, 26 tests passed.

Not tested: other GPUs, other Windows machines, macOS, Linux.

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `running scripts is disabled on this system` | PowerShell execution policy | run scripts as `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` |
| `Python 3.11 or newer was not found` | Python missing or not on PATH | install Python 3.13 from python.org with "Add python.exe to PATH", open a new PowerShell |
| `WinError 206` or "filename or extension is too long" during `pip install` | the repository sits in a deep folder and package paths exceed 260 characters | move the repository to a short path such as `C:\aidetector`, or enable long paths (admin PowerShell): `New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, then restart |
| `OSError: [WinError 126]` mentioning `c10.dll` or `fbgemm.dll` when importing torch | Microsoft Visual C++ runtime missing | install the "Microsoft Visual C++ Redistributable (x64)" from Microsoft, then retry |
| `CUDA error`, `no kernel image`, driver errors, `CUDA out of memory` | GPU driver / CUDA build mismatch or small GPU | `scripts\run.ps1 -Cpu`; for a permanent CPU install delete `.venv` and run `scripts\setup.ps1 -Cpu` |
| `port 8501 is already in use` | another app (or a second copy) is running | close it, or `scripts\run.ps1 -Port 8502` |
| `.venv not found` from `run.ps1` | setup not run in this folder | `scripts\setup.ps1` |
| `model checkpoint checksum` FAIL in `check_env` | damaged or incomplete model download | re-clone or re-download `models\general_native_b.pt`; a file of about 130 bytes is a Git LFS pointer instead of the model |
| `no model found` from `predict.py` | `models\` is empty or the name is wrong | check `models\active_model.json` and `DETECTOR_MODEL` |
| "This file could not be read as an image" in the app | not a JPEG/PNG/WEBP/BMP/TIFF file, or a damaged one | convert the image; other formats (PSD, PDF, GIF, HEIC) are refused on purpose |
| The first analysis is slow | the model is loaded and the GPU warms up once | later images take well under a second |
| Pip fails behind a proxy | network policy | `pip install --proxy http://user:pass@proxy:port ...` or use a network without the proxy |

Still stuck: run `.venv\Scripts\python scripts\check_env.py` and read the `fix:` lines.

## 8. Uninstall

Delete the repository folder. Nothing is installed outside it (the virtual environment is `.venv` inside the folder).
