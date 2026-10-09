"""CLI prediction. All preprocessing/crop/calibration logic lives in detector.analyze() (shared with app.py).

    python predict.py image.jpg [--crop auto|face|none] [--model file.pt] [--json]

Rollback to the original Keras Xception path: set DETECTOR_BACKEND=legacy (needs TensorFlow).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import detector  # noqa: E402
from detector.paths import MODELS  # noqa: E402

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
LEGACY_MODEL_PATH = os.path.join(MODELS, "best_xception.keras")


def legacy_predict(path):
    """Original Keras Xception path, kept only for rollback."""
    import cv2
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    model = tf.keras.models.load_model(LEGACY_MODEL_PATH)
    img = detector.load_rgb(path)
    box = detector.find_face(img)
    region = img.crop(box) if box else img
    arr = np.expand_dims(np.array(region.resize((299, 299)), dtype=np.float32), 0)
    arr = tf.keras.applications.xception.preprocess_input(arr)
    real = float(model.predict(arr, verbose=0)[0][0])
    return dict(real_prob=real, fake_prob=1 - real, model_id="legacy:xception:best_xception.keras",
                crop_policy="face", face_found=box is not None, region_size=region.size, temperature=1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deepfake / AI image detector")
    ap.add_argument("image")
    ap.add_argument("--crop", choices=["auto", "face", "none"], default="auto",
                    help="auto = the active model's validated policy (default)")
    ap.add_argument("--model", default=None, help="checkpoint file name in models/ (default: active model)")
    ap.add_argument("--json", action="store_true", help="print one machine-readable JSON line")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1
    notes = []
    if detector.available(args.model):
        r = detector.analyze(args.image, crop=args.crop, model=args.model)
        out = dict(real_prob=r.real_prob, fake_prob=r.fake_prob, model_id=r.model_id, crop_policy=r.crop_policy,
                   face_found=r.face_found, region_size=list(r.region.size), temperature=r.temperature,
                   kind=r.kind, verdict=r.verdict or None, band=list(r.band) if r.band else None,
                   tile_ai_prob=[round(1 - p, 4) for p in r.tile_real] if r.tile_real else None,
                   flags=[code for code, _ in r.flags], image_size=list(r.image_size))
        notes = [text for _, text in r.flags]
    elif os.path.isfile(LEGACY_MODEL_PATH):
        out = legacy_predict(args.image)
        out["region_size"] = list(out["region_size"])
    else:
        print(f"ERROR: no model found in {MODELS}. Run scripts\\setup.ps1 or see models/README.md.", file=sys.stderr)
        return 1

    out["prediction"] = "REAL" if out["real_prob"] >= 0.5 else "FAKE"
    if args.json:
        print(json.dumps(out))
        return 0
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    general = out.get("kind") == "general"
    crop = out["crop_policy"] + ("" if out["face_found"] is None else
                                 (" (face found)" if out["face_found"] else " (no face found -> full image)"))
    print("\n" + "=" * 62)
    print("       AI IMAGE DETECTOR")
    print("=" * 62)
    print(f"Image            : {args.image}")
    print(f"Model            : {out['model_id']}")
    if general:
        tiles = out["tile_ai_prob"]
        print(f"Image size       : {out['image_size'][0]}x{out['image_size'][1]} px; {len(tiles)} region(s) of 256x256 scored")
        print(f"AI probability   : {out['fake_prob'] * 100:.1f}%   (calibrated; real = {out['real_prob'] * 100:.1f}%)")
        if len(tiles) > 1:
            print(f"Region spread    : {min(tiles) * 100:.0f}% - {max(tiles) * 100:.0f}% AI-probability")
        label = {"likely_fake": "LIKELY AI-GENERATED", "likely_real": "LIKELY REAL",
                 "inconclusive": "INCONCLUSIVE"}.get(out["verdict"], out["prediction"])
        print(f"Verdict          : {label}")
        if out["band"]:
            print(f"Inconclusive band: P(real) between {out['band'][0]:.2f} and {out['band'][1]:.2f}")
    else:
        print(f"Crop policy      : {crop}")
        print(f"Region analyzed  : {out['region_size'][0]}x{out['region_size'][1]} px")
        print(f"Real probability : {out['real_prob'] * 100:.2f}%")
        print(f"Fake probability : {out['fake_prob'] * 100:.2f}%")
        print(f"Prediction       : {'[+] REAL (Authentic)' if out['prediction'] == 'REAL' else '[-] FAKE (AI Generated / Deepfake)'}")
    for n in notes:
        print(f"WARNING          : {n}")
    print("Note             : a calibrated score is an estimate, not proof; see the model card / README for measured")
    print("                   accuracy, per-generator failures and known limits.")
    print("=" * 62 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
