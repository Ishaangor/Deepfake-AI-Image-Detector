# scripts/

Windows helper scripts. Run them from PowerShell in the repository folder.

| command | what it does |
|---|---|
| `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` | Creates `.venv`, installs PyTorch (the CUDA 12.4 wheel if `nvidia-smi` is found, otherwise CPU) and the pinned `requirements.txt`, then runs `check_env.py --smoke`. Add `-Cpu` to force the CPU wheel. |
| `powershell -ExecutionPolicy Bypass -File scripts\run.ps1` | Starts the web UI on http://localhost:8501. Options: `-Cpu` (forces CPU inference), `-Port 8502`. |
| `.venv\Scripts\python scripts\check_env.py [--smoke]` | Checks the Python version, packages, model checksum, calibration file, model card, device and port. `--smoke` also scores one demo image. |
| `powershell -ExecutionPolicy Bypass -File scripts\generate_sbom.ps1` | Regenerates `sbom/sbom.cdx.json` and `sbom/pip-audit.json` (see `docs/SBOM.md`). |

`-ExecutionPolicy Bypass` applies to that one command only and does not change the system policy.
