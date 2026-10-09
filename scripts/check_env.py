"""Environment check for the AI Image Detector. Prints one line per check and exits 1 if anything required fails.

    python scripts/check_env.py            # checks only
    python scripts/check_env.py --smoke    # also scores one demo image end to end (loads the model)
"""
import argparse
import hashlib
import importlib
import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUIRED = ["torch", "torchvision", "timm", "streamlit", "altair", "cv2", "PIL", "numpy", "pandas", "matplotlib"]
OPTIONAL = ["scipy", "sklearn", "pyarrow", "huggingface_hub", "pytest"]
failed = []


def report(ok, what, detail="", fix="", required=True):
    tag = "OK  " if ok else ("FAIL" if required else "WARN")
    print(f"[{tag}] {what}" + (f" - {detail}" if detail else ""))
    if not ok:
        if fix:
            print(f"       fix: {fix}")
        if required:
            failed.append(what)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run one real prediction on a demo image")
    args = ap.parse_args()

    v = sys.version_info
    report(v >= (3, 11), "Python version", f"{v.major}.{v.minor}.{v.micro} (tested: 3.13)",
           "install Python 3.11+ (3.13 tested) from python.org and re-run scripts/setup.ps1")

    for name in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(name)
            report(True, f"package {name}", getattr(mod, "__version__", ""))
        except Exception as e:                                            # noqa: BLE001 - report any import problem
            report(False, f"package {name}", str(e).splitlines()[0],
                   "run scripts/setup.ps1 (or: pip install -r requirements.txt) inside the project's .venv",
                   required=name in REQUIRED)

    models = os.path.join(ROOT, "models")
    try:
        with open(os.path.join(models, "active_model.json"), encoding="utf-8") as f:
            active = json.load(f)["model"]
        report(True, "active model setting", active)
    except Exception as e:                                                # noqa: BLE001
        active = None
        report(False, "active model setting", f"models/active_model.json unreadable ({e})",
               "restore it with: git checkout -- models/active_model.json")

    sums = {}
    try:
        with open(os.path.join(models, "SHA256SUMS"), encoding="utf-8") as f:
            for line in f:
                digest, name = line.split()
                sums[name.lstrip("*")] = digest
    except OSError:
        report(False, "models/SHA256SUMS", "missing", "git checkout -- models/SHA256SUMS", required=False)

    if active:
        ckpt = os.path.join(models, active)
        stem = os.path.splitext(ckpt)[0]
        if not os.path.isfile(ckpt):
            report(False, "model checkpoint", f"{active} not found in models/",
                   "re-clone the repository or run: git checkout -- models/  (the weights are tracked in git)")
        else:
            size = os.path.getsize(ckpt)
            if size < 1_000_000:
                report(False, "model checkpoint", f"{active} is only {size} bytes (Git LFS pointer or partial download?)",
                       "re-download the repository as a full clone")
            elif active in sums:
                report(sha256(ckpt) == sums[active], "model checkpoint checksum", active,
                       "the file differs from the released weights; restore with: git checkout -- models/")
            else:
                report(True, "model checkpoint", f"{active} ({size / 1e6:.1f} MB, no reference checksum)")
        try:
            with open(stem + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            ok = float(meta.get("temperature", 0)) > 0 and (meta.get("band") is None or len(meta["band"]) == 2)
            report(ok, "calibration (temperature + decision band)",
                   f"T={meta.get('temperature'):.3f}, band={meta.get('band')}", "restore models/*.meta.json from git")
        except Exception as e:                                            # noqa: BLE001
            report(False, "calibration metadata", str(e), "restore models/*.meta.json from git")
        report(os.path.isfile(stem + ".card.json"), "measured model card", os.path.basename(stem) + ".card.json",
               "the UI still works but shows the model as unvalidated; restore from git", required=False)

    try:
        import torch
        if os.environ.get("DETECTOR_DEVICE", "").lower() == "cpu":
            report(True, "compute device", "CPU (forced by DETECTOR_DEVICE=cpu)")
        elif torch.cuda.is_available():
            report(True, "compute device", f"CUDA - {torch.cuda.get_device_name(0)}")
        else:
            report(True, "compute device", "CPU (no usable CUDA GPU; analysis takes a few seconds per image)")
    except Exception as e:                                                # noqa: BLE001
        report(False, "compute device", str(e), "reinstall torch with scripts/setup.ps1 -Cpu")

    with socket.socket() as s:
        free = s.connect_ex(("127.0.0.1", 8501)) != 0
    report(free, "port 8501 free", "" if free else "something is already listening (an older Streamlit?)",
           "close the other app, or start on another port: scripts/run.ps1 -Port 8502", required=False)

    if args.smoke and not failed:
        try:
            import detector
            with open(os.path.join(ROOT, "samples", "manifest.json"), encoding="utf-8") as f:
                first = json.load(f)["samples"][0]
            r = detector.analyze(os.path.join(ROOT, "samples", first["file"]))
            report(0.0 <= r.fake_prob <= 1.0, "end-to-end prediction",
                   f"{first['file']}: P(AI) = {r.fake_prob:.3f}, verdict = {r.verdict or 'n/a'} on {detector.device()}")
        except Exception as e:                                            # noqa: BLE001
            report(False, "end-to-end prediction", str(e),
                   "if this mentions CUDA or memory, retry with: $env:DETECTOR_DEVICE='cpu'")

    print()
    if failed:
        print(f"{len(failed)} required check(s) failed: {', '.join(failed)}")
        return 1
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
