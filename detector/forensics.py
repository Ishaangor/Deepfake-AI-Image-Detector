"""Generic file diagnostics shown next to the prediction. The detector does NOT use any of these.

None of them has a validated decision threshold here; ordinary resizing, editing and re-compression change all of them.
They are offered as context ("what kind of file is this?"), never as evidence of AI generation.
"""
import io
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from .inference import jpeg_quality, to_rgb


UPLOAD_FORMATS = ("JPEG", "PNG", "WEBP", "BMP", "TIFF")   # the JPEG opener also handles multi-picture (MPO) phone JPEGs


def open_upload(data: bytes) -> Image.Image:
    """Open untrusted uploaded bytes with only the decoders the UI advertises. Pillow sniffs content rather than trusting
    the file extension, so without this a renamed PSD/PDF/font file would reach rarely used (and historically buggy)
    decoders. Anything else raises PIL.UnidentifiedImageError."""
    return Image.open(io.BytesIO(data), formats=UPLOAD_FORMATS)


def file_info(data: bytes) -> dict:
    """Container facts read from the uploaded bytes: format, colour mode, pixel size, file size, EXIF count, JPEG quality."""
    with open_upload(data) as im:
        exif = len(im.getexif()) if hasattr(im, "getexif") else 0
        return dict(format=im.format, mode=im.mode, size=im.size, bytes=len(data), exif_fields=exif,
                    jpeg_q=jpeg_quality(im))


def ela(pil: Image.Image, quality: int = 90, max_side: int = 1024) -> Image.Image:
    """Error Level Analysis: |image - JPEG(image, quality)|, contrast-stretched.

    Bright areas changed more when re-saved. This reflects the image's compression history and its content (edges and
    fine texture are always brighter). It is not a detector of AI generation."""
    pil = to_rgb(pil)
    pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    diff = ImageChops.difference(pil, Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    top = max(e[1] for e in diff.getextrema())
    return ImageEnhance.Brightness(diff).enhance(255.0 / max(top, 1))


def spectrum(pil: Image.Image, size: int = 256) -> np.ndarray:
    """Log-magnitude 2-D Fourier spectrum of the grayscale image resized to `size` px, as a uint8 RGB (viridis) array.

    The centre is low frequency. Regular grids of bright dots can come from periodic patterns - upsampling layers of
    some generators, but equally from resizing, JPEG blocks or textures in the scene."""
    import cv2
    g = np.asarray(to_rgb(pil).convert("L").resize((size, size)), dtype=np.float32)
    mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1e-8)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def blend_heatmap(base: np.ndarray, heat: np.ndarray, examined: Optional[np.ndarray], opacity: float) -> np.ndarray:
    """Grad-CAM heat map over the analysed image with a user-chosen opacity; pixels no tile examined are dimmed."""
    from .gradcam import overlay
    out = overlay(base, heat, alpha=float(opacity))
    if examined is not None:
        out = np.where(examined[..., None], out, (base * 0.35).astype(np.uint8))
    return out
