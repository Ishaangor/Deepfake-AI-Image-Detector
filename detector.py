"""Shared inference for the fine-tuned PyTorch detector.

Both predict.py and app.py call analyze(), so decoding, color handling, crop policy, resizing,
test-time augmentation and temperature scaling are identical in every entry point.

Model selection (rollback = change one line):
  1. env DETECTOR_BACKEND=legacy      -> available() is False; callers fall back to the old Keras/Swin path
  2. env DETECTOR_MODEL=<file.pt>     -> use that checkpoint from models/
  3. models/active_model.json         -> {"model": "<file.pt>"}
  4. default                          -> best_finetuned.pt (the original checkpoint)

Each checkpoint <name>.pt may have <name>.meta.json:
  {"temperature": float, "crop_policy": "none" | "face", "arch": str, "img": int}
Without it, models/calibration.json (original checkpoint) is used and crop_policy defaults to "none".

Two kinds of checkpoint exist:
  face    : the original face-trained EfficientNet-B0 family (single resized image, optional Haar face crop)
  general : produced by train_general.py (key "view" in the checkpoint). Native-resolution 256 px tiles (or one global
            view) are scored by general.py; the tile logits are averaged, divided by the calibration temperature and the
            result is mapped to a verdict with the decision band stored in the meta file ("band": [lo, hi]).
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

MODELS_DIR = "models"
DEFAULT_MODEL = "best_finetuned.pt"
ACTIVE_FILE = os.path.join(MODELS_DIR, "active_model.json")
LEGACY_CALIB = os.path.join(MODELS_DIR, "calibration.json")
FACE_MARGIN = 0.15
MIN_FACE = 60

_cache = {}


# ---------------------------------------------------------------- decoding / color handling
def to_rgb(im: Image.Image) -> Image.Image:
    """Standardised decode to 8-bit RGB: EXIF orientation, palette, 16-bit, CMYK, grayscale and alpha.

    Images with real transparency are composited on white; fully opaque alpha channels (e.g. the
    grayscale+alpha PNGs in the training data) are simply dropped. Grayscale stays grayscale
    (replicated to 3 channels) -- callers must not assume a color image.
    """
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = arr / (257.0 if arr.max() > 255 else 1.0)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    if im.mode == "F":
        return Image.fromarray(np.clip(np.asarray(im), 0, 255).astype(np.uint8)).convert("RGB")
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA", "PA"):
        rgba = im.convert("RGBA")
        alpha = rgba.getchannel("A")
        if alpha.getextrema()[0] < 255:
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, rgba).convert("RGB")
        return rgba.convert("RGB")
    return im.convert("RGB")


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return to_rgb(im)


# ---------------------------------------------------------------- face crop policy
def _cascade():
    if "cascade" not in _cache:
        import cv2
        _cache["cascade"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cache["cascade"]


def find_face(pil: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Largest Haar-cascade face expanded by FACE_MARGIN, as (x1, y1, x2, y2); None if no face."""
    import cv2
    gray = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    pw, ph = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    return max(0, x - pw), max(0, y - ph), min(pil.size[0], x + w + pw), min(pil.size[1], y + h + ph)


# ---------------------------------------------------------------- model registry
def active_model_name() -> str:
    if os.environ.get("DETECTOR_MODEL"):
        return os.environ["DETECTOR_MODEL"]
    try:
        with open(ACTIVE_FILE) as f:
            return json.load(f)["model"]
    except (OSError, KeyError, ValueError):
        return DEFAULT_MODEL


def model_path(name: Optional[str] = None) -> str:
    return os.path.join(MODELS_DIR, name or active_model_name())


def available(name: Optional[str] = None) -> bool:
    if os.environ.get("DETECTOR_BACKEND", "").lower() == "legacy":
        return False
    return os.path.isfile(model_path(name))


def _meta(path: str, ck: dict) -> dict:
    meta = {"temperature": 1.0, "crop_policy": "none", "arch": ck["arch"], "img": ck["img"],
            "kind": "general" if "view" in ck else "face", "view": ck.get("view"), "band": None,
            "explain_layer": "bn2"}
    side = os.path.splitext(path)[0] + ".meta.json"
    src = side if os.path.isfile(side) else (LEGACY_CALIB if os.path.basename(path) == DEFAULT_MODEL else None)
    if src:
        with open(src) as f:
            meta.update({k: v for k, v in json.load(f).items() if k in meta and k not in ("kind", "view")})
    return meta


def _load(name: Optional[str] = None) -> dict:
    path = model_path(name)
    if path not in _cache:
        import timm
        import torch
        from torchvision import transforms as T
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(path, map_location=dev)
        meta = _meta(path, ck)
        model = timm.create_model(ck["arch"], pretrained=False, num_classes=1)
        model.load_state_dict(ck["state"])
        model.to(dev).eval()
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:8]
        tf = T.Compose([
            T.Resize((meta["img"], meta["img"])), T.ToTensor(),
            T.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD),
        ])
        _cache[path] = dict(model=model, dev=dev, tf=tf, meta=meta, view=ck.get("view"),
                            model_id=f"{ck['arch']}:{os.path.basename(path)}@{sha}")
    return _cache[path]


def default_crop_policy(name: Optional[str] = None) -> str:
    return _load(name)["meta"]["crop_policy"]


# ---------------------------------------------------------------- inference
@dataclass
class Result:
    real_prob: float
    fake_prob: float
    model_id: str
    crop_policy: str                       # policy actually applied: "none" or "face"
    face_found: Optional[bool]             # None when no detection was attempted
    box: Optional[Tuple[int, int, int, int]]
    region: Image.Image                    # the exact pixels fed to the network (before resize)
    temperature: float
    kind: str = "face"                     # "face" (original family) or "general"
    verdict: str = ""                      # general: "likely_real" | "likely_fake" | "inconclusive"
    band: Optional[Tuple[float, float]] = None   # calibrated P(real) thresholds [lo, hi] used for the verdict
    tile_real: Optional[list] = None       # calibrated P(real) of each examined tile (general)
    boxes: Optional[list] = None           # tile boxes (x0, y0, x1, y1) in the prepared image (general)
    prepared_size: Optional[Tuple[int, int]] = None
    image_size: Tuple[int, int] = (0, 0)
    flags: list = field(default_factory=list)    # [(code, text)] input-quality / applicability notes


_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
             29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
             121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def jpeg_quality(im) -> Optional[float]:
    """IJG-scale quality estimate from the luminance quantisation table of a still-JPEG PIL image, else None."""
    qt = getattr(im, "quantization", None)
    if not qt or 0 not in qt:
        return None
    s = 100.0 * sum(qt[0]) / sum(_STD_LUMA)
    return float(min(max((200 - s) / 2 if s <= 100 else 5000 / s, 1), 100))


def quality_flags(pil: Image.Image, jpeg_q: Optional[float], kind: str) -> list:
    """Facts about the INPUT that lower or limit the reliability of any verdict. Thresholds are descriptive, not tuned."""
    w, h = pil.size
    out = []
    if kind == "face":
        out.append(("face_only_model", "This model was trained on face photographs only. Its output is not meaningful "
                                       "for other kinds of images."))
    if min(w, h) < 256:
        out.append(("low_res", f"Image is only {w}x{h} px. It was enlarged to 256 px for analysis; the detector was trained "
                               "on native-resolution crops of 256 px or more, so this result is unreliable."))
    if jpeg_q is not None and jpeg_q < 60:
        out.append(("heavy_jpeg", f"Heavy JPEG compression (estimated quality ~{jpeg_q:.0f}). Compression removes the fine "
                                  "pixel statistics detectors rely on; accuracy drops."))
    a = np.asarray(pil.resize((64, 64)), dtype=np.int16)
    if np.abs(a[..., 0] - a[..., 1]).max() <= 2 and np.abs(a[..., 1] - a[..., 2]).max() <= 2:
        out.append(("grayscale", "Image is grayscale. The detector saw mostly colour images in training."))
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _verdict(p_real: float, band) -> str:
    lo, hi = (band if band else (0.5, 0.5))
    return "likely_fake" if p_real < lo else ("likely_real" if p_real > hi else "inconclusive")


def _jpeg_q_of(image) -> Optional[float]:
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as im:
            return jpeg_quality(im)
    return jpeg_quality(image)


def analyze(image, crop: str = "auto", model: Optional[str] = None) -> Result:
    """image: PIL image or path. crop: "auto" (model's validated policy), "face" or "none" (face-kind models only)."""
    import torch
    q = _jpeg_q_of(image)
    pil = load_rgb(image) if isinstance(image, (str, os.PathLike)) else to_rgb(image)
    c = _load(model)
    meta = c["meta"]
    if meta["kind"] == "general":
        import general
        ts, boxes, size = general.tiles(pil, c["view"])
        z = general.tile_logits(c["model"], ts, c["dev"], flip=True)
        T = meta["temperature"]
        real = float(_sigmoid(z.mean() / T))
        tile_real = _sigmoid(z / T).tolist()
        band = tuple(meta["band"]) if meta["band"] else None
        flags = quality_flags(pil, q, "general")
        verdicts = {_verdict(p, band) for p in tile_real}
        if "likely_fake" in verdicts and "likely_real" in verdicts:
            flags.append(("tiles_disagree", "Different parts of the image point in opposite directions "
                                            f"({len(tile_real)} regions examined)."))
        covered = len(ts) * 256 * 256 / max(size[0] * size[1], 1)
        if c["view"] == "native" and covered < 0.6:
            flags.append(("partial_coverage", f"Only about {covered:.0%} of the pixels were examined "
                                              f"({len(ts)} crops of 256 px); the rest of the image was not looked at."))
        return Result(real, 1.0 - real, c["model_id"], "none", None, None, pil, T, kind="general",
                      verdict=_verdict(real, band), band=band, tile_real=tile_real, boxes=boxes, prepared_size=size,
                      image_size=pil.size, flags=flags)
    policy = meta["crop_policy"] if crop == "auto" else crop
    if policy not in ("none", "face"):
        raise ValueError(f"unknown crop policy {policy!r}")
    box, found, region = None, None, pil
    if policy == "face":
        box = find_face(pil)
        found = box is not None
        if found:
            region = pil.crop(box)
    x = c["tf"](region).unsqueeze(0).to(c["dev"])
    with torch.no_grad():
        z = c["model"](torch.cat([x, torch.flip(x, dims=[3])])).squeeze(1).float().mean() / meta["temperature"]
    real = float(torch.sigmoid(z))
    return Result(real, 1.0 - real, c["model_id"], policy, found, box, region, meta["temperature"], kind="face",
                  image_size=pil.size, flags=quality_flags(pil, q, "face"))


# ---------------------------------------------------------------- explanation (model attention, not proof)
@dataclass
class Explanation:
    overlay: np.ndarray                    # uint8 RGB of the prepared image with the heat map blended in
    heat: np.ndarray                       # float [0,1] at overlay size; 0 where nothing was examined
    examined: np.ndarray                   # bool mask of pixels covered by at least one analysed tile
    layer: str
    evidence_for: str                      # "real" or "fake": the class the map explains
    base: np.ndarray                       # uint8 RGB prepared image without overlay


def explain(image, model: Optional[str] = None, evidence_for: str = "auto", max_side: int = 1024) -> Explanation:
    """Grad-CAM of the model's own logit for the chosen class, stitched over every analysed tile.

    The map shows which input regions pushed the model toward that class. It does NOT show that a region is edited or
    generated. Overlapping tiles are averaged and the map is normalised by its global maximum.
    """
    import cv2
    import explain as X
    pil = load_rgb(image) if isinstance(image, (str, os.PathLike)) else to_rgb(image)
    c = _load(model)
    meta, net, dev = c["meta"], c["model"], c["dev"]
    if evidence_for == "auto":
        evidence_for = "real" if analyze(pil, model=model).real_prob >= 0.5 else "fake"
    sign = 1.0 if evidence_for == "real" else -1.0
    layer = dict(net.named_modules())[meta["explain_layer"]]
    if meta["kind"] == "general":
        import general
        ts, boxes, _ = general.tiles(pil, c["view"])
        base = general.prepare(pil, c["view"])
        sub = [(b, general._TF(t).unsqueeze(0).to(dev)) for b, t in zip(boxes, ts)]
    else:
        base = pil
        sub = [((0, 0, base.size[0], base.size[1]), c["tf"](base).unsqueeze(0).to(dev))]
    W, H = base.size
    acc, cnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for (x0, y0, x1, y1), x in sub:
        cam, _ = X.gradcam(net, x, layer, sign, raw=True)
        if cam.shape != (y1 - y0, x1 - x0):
            cam = cv2.resize(cam, (x1 - x0, y1 - y0))
        acc[y0:y1, x0:x1] += cam
        cnt[y0:y1, x0:x1] += 1
    heat = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)
    heat = heat / max(float(heat.max()), 1e-12)
    arr = np.asarray(base)
    s = min(1.0, max_side / max(W, H))
    if s < 1:
        arr = cv2.resize(arr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        heat = cv2.resize(heat, (arr.shape[1], arr.shape[0]))
        cnt = cv2.resize(cnt, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
    over = X.overlay(arr, heat)
    over = np.where((cnt > 0)[..., None], over, (arr * 0.35).astype(np.uint8))      # dim the unexamined pixels
    return Explanation(over, heat, cnt > 0, meta["explain_layer"], evidence_for, arr)
