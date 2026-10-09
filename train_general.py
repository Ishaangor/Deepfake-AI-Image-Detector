"""Train a general real-vs-AI-image detector on the Community Forensics cache (cf_data / cf_build).

    python train_general.py NAME [--view native|glob] [--arch efficientnet_b0] [--epochs 8] [--lr 2e-4]

* trains on split 'train' only (generators, camera devices and images disjoint from 'val' and 'calib');
* augmentation is class-independent (same chain for reals and fakes);
* the epoch / candidate is chosen on 'val' (UNSEEN generators and cameras) with a fixed composite:
  0.5 * clean AUC + 0.5 * mean AUC over {jpeg50, down128, blur1.0};
* 'calib' (temperature) and the independent CommunityForensics-Eval benchmark are never touched here.
Output: models/general_<NAME>.pt  (keys arch, img, view, state, ...), artifacts/train_general_<NAME>.json
"""
import argparse
import io
import json
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T

import cf_data
import perturb

NORM = T.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)
TO_T = T.Compose([T.ToTensor(), NORM])
CROP = 224
AUG = dict(p_res=.25, p_blur=.15, p_jpeg=.35, p_gray=.08, p_color=.5)


class TrainTF:
    def __init__(self, view):
        self.view = view
        self.glob_crop = T.RandomResizedCrop(CROP, scale=(.6, 1.0), ratio=(.9, 1.1))
        self.color = T.ColorJitter(.3, .3, .3, .03)

    def __call__(self, im):
        c = AUG
        if self.view == "native":                                # pixels must stay un-resized for the crop itself
            x, y = random.randint(0, im.width - CROP), random.randint(0, im.height - CROP)
            im = im.crop((x, y, x + CROP, y + CROP))
        else:
            im = self.glob_crop(im)
        if random.random() < c["p_res"]:                         # resolution loss with random kernels
            t = random.randint(96, CROP - 16)
            im = im.resize((t, t), random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX]))
            im = im.resize((CROP, CROP), random.choice([Image.BILINEAR, Image.BICUBIC]))
        if random.random() < c["p_blur"]:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(.3, 1.5)))
        if random.random() < .5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < c["p_color"]:
            im = self.color(im)
        if random.random() < c["p_gray"]:
            im = im.convert("L").convert("RGB")
        if random.random() < c["p_jpeg"]:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=random.randint(35, 95))
            im = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        return TO_T(im)


# Source mixture inside each class, fixed before any benchmark-v2 result existed. Motivation (from the first CompEval
# result and the data audit): web-style reals and commercial-generator fakes are scarce in the Small pool, so they are
# up-weighted instead of letting the 20k SD-fine-tune fakes / 18k COCO-FFHQ-LHQ-VISION reals dominate.
MIX_REAL = dict(coco=.10, vision=.08, lhq=.08, ffhq=.08, openimages=.21, ce_laion=.35, ce_imagenet=.10)
MIX_FAKE = dict(fake_Systematic=.30, fake_Manual=.20, ce_fake=.50)

VAL_CONDS = ("clean", "jpeg50", "down128", "blur1.0")


@torch.no_grad()
def val_logits(model, ds, fn, dev, bs=64):
    model.eval()
    out = []
    for i in range(0, len(ds), bs):
        x = torch.stack([TO_T(fn(ds.image(j))) for j in range(i, min(i + bs, len(ds)))]).to(dev)
        with torch.autocast(dev, dtype=torch.float16):
            out.append(model(x).reshape(-1).float().cpu())
    return torch.cat(out).numpy()


def composite(model, ds, dev):
    y = ds.y
    aucs = {c: float(roc_auc_score(y, val_logits(model, ds, perturb.SEEN[c], dev))) for c in VAL_CONDS}
    return 0.5 * aucs["clean"] + 0.5 * float(np.mean([aucs[c] for c in VAL_CONDS[1:]])), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--view", default="native", choices=["native", "glob"])
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--val_n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix", default="source", choices=["source", "natural"])
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va = cf_data.load_meta("train"), cf_data.load_meta("val")
    va = va.groupby("y_real", group_keys=False).apply(lambda g: g.sample(min(len(g), a.val_n // 2), random_state=1))
    print(f"{a.name}: view={a.view} arch={a.arch} train={len(tr)} (real {int(tr.y_real.sum())}) "
          f"val={len(va)} (real {int(va.y_real.sum())})", flush=True)
    val_ds = cf_data.CFDataset(va, a.view, TO_T)
    if a.mix == "natural":
        w = np.where(tr.y_real == 1, 0.5 / tr.y_real.sum(), 0.5 / (1 - tr.y_real).sum())
    else:                                                     # fixed source mixture inside each class (see MIX_*)
        subset = tr["subset"].fillna("").astype(str) if "subset" in tr else pd.Series("", index=tr.index)
        bucket = np.where(tr.role == "fake", "fake_" + subset, tr.role)
        cnt = pd.Series(bucket).value_counts()
        mix = {**MIX_REAL, **MIX_FAKE}
        tot = {c: sum(mix[k] for k in cnt.index if (k in MIX_REAL) == (c == "real")) for c in ("real", "fake")}
        share = {k: mix[k] / tot["real" if k in MIX_REAL else "fake"] for k in cnt.index}
        w = np.array([0.5 * share[b] / cnt[b] for b in bucket])
        print("sampling shares:", {k: round(0.5 * v, 3) for k, v in share.items()}, flush=True)
    sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
    dl = DataLoader(cf_data.CFDataset(tr, a.view, TrainTF(a.view)), batch_size=a.batch, sampler=sampler,
                    num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0, drop_last=True)
    model = timm.create_model(a.arch, pretrained=True, num_classes=1, drop_rate=0.2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    log = {"args": vars(a), "aug": AUG, "n_train": len(tr), "epochs": []}
    best, out_pt = -1.0, os.path.join("models", f"general_{a.name}.pt")
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for i, (x, y, _) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x).reshape(-1).float(), y * 0.98 + 0.01)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if i % 100 == 0:
                print(f"  ep{ep + 1} it{i}/{len(dl)} loss={loss.item():.4f}", flush=True)
        comp, aucs = composite(model, val_ds, dev)
        print(f"epoch {ep + 1}/{a.epochs} loss={tot / len(dl):.4f} val composite={comp:.4f} " +
              " ".join(f"{k}={v:.3f}" for k, v in aucs.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        log["epochs"].append({"epoch": ep + 1, "loss": tot / len(dl), "composite": comp, "aucs": aucs})
        if comp > best:
            best = comp
            torch.save({"arch": a.arch, "img": 256, "view": a.view, "state": model.state_dict(), "epoch": ep + 1,
                        "val_composite": comp, "name": a.name, "train_crop": CROP}, out_pt)
            log["best_epoch"], log["best_composite"] = ep + 1, comp
        json.dump(log, open(os.path.join("artifacts", f"train_general_{a.name}.json"), "w"), indent=1)
    print(f"done. best epoch={log['best_epoch']} composite={best:.4f} -> {out_pt}")


if __name__ == "__main__":
    main()
