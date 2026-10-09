"""Inference for the general-purpose (non-face-specific) detectors trained by train_general.py.

A checkpoint stores which *view* it was trained on:
  native : 256x256 crops taken at native resolution (never resized, 8-aligned like the training cache). An image is
           covered by up to MAX_TILES crops spread over the image; images smaller than 256 px are upscaled first
           (flagged as low-resolution because the model has never seen genuinely small images at native scale).
  glob   : whole image, shorter side resized to 256, centre crop.
Per-tile logits (logit of P(real), project convention) are averaged; their spread is an uncertainty signal.
"""
import math

import timm
import torch
from PIL import Image
from torchvision import transforms as T

TILE = 256
MAX_TILES = 9
_TF = T.Compose([T.ToTensor(), T.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)])


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=1)
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck


def prepare(im, view):
    """The image as the network will see it before tiling: 'glob' -> shorter side resized to 256; 'native' -> unchanged
    unless the shorter side is below 256 px, in which case it is upscaled (bicubic) to 256."""
    W, H = im.size
    if view == "glob":
        s = TILE / min(W, H)
        return im.resize((max(TILE, round(W * s)), max(TILE, round(H * s))), Image.BICUBIC)
    if min(W, H) < TILE:
        s = TILE / min(W, H)
        return im.resize((max(TILE, math.ceil(W * s)), max(TILE, math.ceil(H * s))), Image.BICUBIC)
    return im


def tiles(im, view, max_tiles=MAX_TILES):
    """Return (list of 256x256 PIL tiles, list of boxes (x0, y0, x1, y1) in the prepared image, prepared image size)."""
    g = prepare(im, view)
    W, H = g.size
    if view == "glob":
        x, y = (W - TILE) // 2, (H - TILE) // 2
        return [g.crop((x, y, x + TILE, y + TILE))], [(x, y, x + TILE, y + TILE)], g.size
    nx, ny = min(math.ceil(W / TILE), 3), min(math.ceil(H / TILE), 3)
    xs = [8 * int((W - TILE) * i / (nx - 1) // 8) for i in range(nx)] if nx > 1 else [8 * ((W - TILE) // 16)]
    ys = [8 * int((H - TILE) * i / (ny - 1) // 8) for i in range(ny)] if ny > 1 else [8 * ((H - TILE) // 16)]
    boxes = [(x, y, x + TILE, y + TILE) for y in ys for x in xs][:max_tiles]
    return [g.crop(b) for b in boxes], boxes, g.size


@torch.no_grad()
def tile_logits(model, ts, device="cpu", flip=True):
    x = torch.stack([_TF(t) for t in ts]).to(device)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        z = model(x).reshape(-1).float()
        if flip:
            z = (z + model(torch.flip(x, [3])).reshape(-1).float()) / 2
    return z.cpu().numpy()


def score(model, ck, im, device="cpu", flip=True):
    """Mean logit over tiles + spread. `im` is an RGB PIL image."""
    ts, boxes, size = tiles(im, ck["view"])
    z = tile_logits(model, ts, device, flip)
    return dict(logit=float(z.mean()), tile_logits=z, std=float(z.std()) if len(z) > 1 else 0.0, boxes=boxes,
                scaled_size=size, n_tiles=len(ts))
