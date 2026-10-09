"""Deterministic perturbation suite, fixed in advance (never tuned on the test set).

Each condition maps a native-resolution RGB PIL image to the 256x256 RGB image the network sees.
  SEEN      : families used by training augmentation; also used on val_sel (selection) and calib
  HARSHER   : same families, stronger than anything in training (test-only)
  NEW_FAMILY: operations that never appear in training augmentation (test-only)
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

S = 256


def _base(im):
    return im.resize((S, S), Image.BILINEAR)


def _jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _webp(im, q):
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def _downup(im, side, down=Image.BILINEAR, up=Image.BILINEAR):
    return _base(im).resize((side, side), down).resize((S, S), up)


def _color(im):
    im = ImageEnhance.Brightness(_base(im)).enhance(0.75)
    im = ImageEnhance.Contrast(im).enhance(1.3)
    return ImageEnhance.Color(im).enhance(0.6)


def _noise(im, sigma=6.0):
    a = np.asarray(_base(im), dtype=np.float32)
    n = np.random.RandomState(0).normal(0, sigma, a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


SEEN = {
    "clean": _base,
    "jpeg50": lambda im: _jpeg(_base(im), 50),
    "down128": lambda im: _downup(im, 128),
    "down96": lambda im: _downup(im, 96, Image.BICUBIC, Image.BICUBIC),
    "blur1.0": lambda im: _base(im).filter(ImageFilter.GaussianBlur(1.0)),
    "gray": lambda im: _base(im).convert("L").convert("RGB"),
    "color": _color,
}
HARSHER = {
    "jpeg30": lambda im: _jpeg(_base(im), 30),
    "down64": lambda im: _downup(im, 64),
    "blur2.0": lambda im: _base(im).filter(ImageFilter.GaussianBlur(2.0)),
    "jpeg30+down128": lambda im: _jpeg(_downup(im, 128), 30),
}
NEW_FAMILY = {
    "webp50": lambda im: _webp(_base(im), 50),
    "down80_nearest": lambda im: _downup(im, 80, Image.NEAREST, Image.NEAREST),
    "noise6": _noise,
}
ALL = {**SEEN, **HARSHER, **NEW_FAMILY}
