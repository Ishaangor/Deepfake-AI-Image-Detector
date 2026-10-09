"""Model-attention maps and faithfulness checks for the timm CNN detectors.

A map produced here explains which input regions the *model* used for its output logit z = log-odds(real). It is not
evidence that a region was manipulated. All functions take a normalised tensor batch of one image (1,3,H,W) and a
model that returns one logit per image (positive = real).

    gradcam(model, x, layer, sign)      Grad-CAM on `layer` for sign*z  (sign=+1 real evidence, -1 fake evidence)
    occlusion(model, x, ...)            perturbation map, independent of gradients (used to cross-check Grad-CAM)
    deletion_insertion(...)             faithfulness curves vs random / reverse orderings
    randomization_check(...)            maps must change when trained weights are re-initialised
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F


def _logit(model, x):
    return model(x).reshape(-1)


def gradcam(model, x, layer, sign=1.0, raw=False):
    """Grad-CAM map in [0, 1], shape (H, W), for sign*z. `layer` is an nn.Module inside `model`.

    raw=True returns the un-normalised positive map (comparable between tiles of one image) instead of min-max scaling."""
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    model.eval()
    try:
        x = x.clone().requires_grad_(True)
        z = _logit(model, x)[0]
        model.zero_grad(set_to_none=True)
        (sign * z).backward()
    finally:
        h1.remove()
        h2.remove()
    a, g = store["a"].detach(), store["g"].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    if raw:
        return cam.cpu().numpy(), float(z.detach())
    cam = cam - cam.min()
    return (cam / cam.max().clamp_min(1e-12)).cpu().numpy(), float(z.detach())


@torch.no_grad()
def occlusion(model, x, patch=32, stride=16, sign=1.0):
    """Drop in sign*z when a patch is replaced by the image's mean colour; mapped back to pixels, rescaled to [0, 1]."""
    model.eval()
    _, _, H, W = x.shape
    base = float(sign * _logit(model, x)[0])
    fill = x.mean(dim=(2, 3), keepdim=True)
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    batch, pos = [], []
    for y in ys:
        for xx in xs:
            xo = x.clone()
            xo[:, :, y:y + patch, xx:xx + patch] = fill
            batch.append(xo)
            pos.append((y, xx))
    drops = []
    for i in range(0, len(batch), 32):
        drops.append(base - sign * _logit(model, torch.cat(batch[i:i + 32])).float())
    drops = torch.cat(drops).cpu().numpy()
    acc, cnt = np.zeros((H, W)), np.zeros((H, W))
    for (y, xx), d in zip(pos, drops):
        acc[y:y + patch, xx:xx + patch] += d
        cnt[y:y + patch, xx:xx + patch] += 1
    m = acc / np.maximum(cnt, 1)
    m = np.clip(m, 0, None)
    return m / max(m.max(), 1e-12)


def _grid_scores(cam, cell):
    """Mean map value per cell x cell block, flattened."""
    H, W = cam.shape
    gh, gw = H // cell, W // cell
    return cam[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3)).reshape(-1), gh, gw


@torch.no_grad()
def deletion_insertion(model, x, cam, sign=1.0, cell=16, steps=12, seed=0):
    """Faithfulness curves for sign*z.

    deletion : blank cells in order of decreasing map value; a faithful map makes sign*z fall fastest.
    insertion: start blank, reveal cells in decreasing order; a faithful map makes sign*z rise fastest.
    Each is compared with a random cell order and with the reversed (least-important-first) order.
    Returns dict of curves (lists of sign*z over `steps+1` fractions).
    """
    model.eval()
    _, _, H, W = x.shape
    scores, gh, gw = _grid_scores(cam, cell)
    n = len(scores)
    rng = np.random.RandomState(seed)
    orders = {"map": np.argsort(-scores, kind="stable"), "reverse": np.argsort(scores, kind="stable"),
              "random": rng.permutation(n)}
    fill = x.mean(dim=(2, 3), keepdim=True).expand_as(x)

    def mask_for(idx):
        m = torch.zeros(gh * gw, device=x.device)
        m[idx] = 1
        m = m.reshape(1, 1, gh, gw)
        m = F.interpolate(m, scale_factor=cell, mode="nearest")
        out = torch.zeros(1, 1, H, W, device=x.device)
        out[:, :, :m.shape[2], :m.shape[3]] = m
        return out

    fracs = np.linspace(0, 1, steps + 1)
    res = {"fracs": fracs.tolist()}
    for kind in ("deletion", "insertion"):
        for name, order in orders.items():
            xs = []
            for f in fracs:
                k = int(round(f * n))
                m = mask_for(torch.as_tensor(order[:k], device=x.device, dtype=torch.long))
                xs.append(x * (1 - m) + fill * m if kind == "deletion" else x * m + fill * (1 - m))
            z = sign * _logit(model, torch.cat(xs)).float().cpu().numpy()
            res[f"{kind}_{name}"] = z.tolist()
    return res


def curve_area(curve):
    """Mean of the curve (trapezoid over [0,1]) - larger = higher sign*z on average."""
    c = np.asarray(curve, dtype=float)
    return float(np.sum((c[1:] + c[:-1]) / 2) / (len(c) - 1))


def randomization_check(model, x, layer_name, sign=1.0, seed=0):
    """Spearman correlation between the trained model's Grad-CAM and the Grad-CAM of the same architecture with
    re-initialised weights. A map that really depends on learned weights should be weakly correlated."""
    from scipy.stats import spearmanr
    layer = dict(model.named_modules())[layer_name]
    cam_t, _ = gradcam(model, x, layer, sign)
    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    for m in rnd.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    cam_r, _ = gradcam(rnd, x, dict(rnd.named_modules())[layer_name], sign)
    rho = spearmanr(cam_t.reshape(-1)[::7], cam_r.reshape(-1)[::7]).correlation
    return float(rho)


def overlay(image_rgb, cam, alpha=0.45):
    """Blend a [0,1] map (resized to the image) as a jet heatmap over a uint8 RGB array. Returns uint8 RGB."""
    import matplotlib
    cmap = matplotlib.colormaps["jet"]
    H, W = image_rgb.shape[:2]
    cam_t = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]
    cam_r = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = (cmap(cam_r)[..., :3] * 255).astype(np.float32)
    a = alpha * cam_r[..., None]          # transparent where the model paid little attention
    return np.clip(image_rgb * (1 - a) + heat * a, 0, 255).astype(np.uint8)
