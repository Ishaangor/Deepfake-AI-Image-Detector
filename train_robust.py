"""Robustness fine-tuning experiment, started from the existing checkpoint (which is never overwritten).

    python train_robust.py A_mild      # or B_full

* trains on splits_v2 'train' only; augmentation is class-independent (reals and fakes get the same
  chain), so format / resolution / sharpness / color-vs-grayscale cannot become a class cue by themselves
* the epoch is selected on 'val_sel' with a fixed deterministic composite (perturb.SEEN): half the weight on
  clean AUC, half on the mean AUC over the other seen conditions
* 'calib' (temperature) and 'test' (final report) are never touched here
"""
import io
import json
import os
import random
import sys
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import detector
import perturb
import splits

INIT = os.path.join("models", "best_finetuned.pt")
VARIANTS = {
    "A_mild": dict(p_res=.25, p_blur=.15, p_jpeg=.35, p_gray=.10, p_color=.5),
    "B_full": dict(p_res=.50, p_blur=.30, p_jpeg=.50, p_gray=.25, p_color=.8),
}
IMG = 256
EPOCHS = int(os.environ.get("EPOCHS", 6))
BATCH = int(os.environ.get("BATCH", 32))
LR = float(os.environ.get("LR", 1e-4))
WORKERS = int(os.environ.get("WORKERS", 3))
SEED = int(os.environ.get("SEED", 42))

_norm = T.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)
_to_tensor = T.Compose([T.ToTensor(), _norm])


class TrainTF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.crop = T.RandomResizedCrop(IMG, scale=(.5, 1.0), ratio=(.9, 1.1))
        self.color = T.ColorJitter(.3, .3, .3, .03)

    def __call__(self, im):
        c = self.cfg
        im = self.crop(im)
        if random.random() < c["p_res"]:                       # resolution loss, random kernels
            t = random.randint(80, 224)
            im = im.resize((t, t), random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX]))
            im = im.resize((IMG, IMG), random.choice([Image.BILINEAR, Image.BICUBIC]))
        if random.random() < c["p_blur"]:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(.3, 1.5)))
        if random.random() < .5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < c["p_color"]:
            im = self.color(im)
        if random.random() < c["p_gray"]:                       # color must not be a class cue
            im = im.convert("L").convert("RGB")
        if random.random() < c["p_jpeg"]:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=random.randint(35, 95))
            im = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        return _to_tensor(im)


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(detector.load_rgb(p)), torch.tensor(float(y))


def logits(model, imgs, fn, dev):
    model.eval()
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([_to_tensor(fn(im)) for im in imgs[i:i + 64]]).to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16):
            out.append(model(x).squeeze(1).float().cpu())
    return torch.cat(out).numpy()


def composite(model, imgs, ys, dev):
    aucs = {name: roc_auc_score(ys, logits(model, imgs, fn, dev)) for name, fn in perturb.SEEN.items()}
    others = [v for k, v in aucs.items() if k != "clean"]
    return 0.5 * aucs["clean"] + 0.5 * float(np.mean(others)), aucs


def main(variant):
    cfg = VARIANTS[variant]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp, meta = splits.load()
    print(f"variant={variant} cfg={cfg} train={len(sp['train'])} val_sel={len(sp['val_sel'])} epochs={EPOCHS} lr={LR}", flush=True)
    val_imgs = [detector.load_rgb(p) for p, _ in sp["val_sel"]]
    val_y = np.array([y for _, y in sp["val_sel"]])

    ck = torch.load(INIT, map_location=dev)
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=1, drop_rate=0.3)
    model.load_state_dict(ck["state"])
    model.to(dev)

    base_comp, base_aucs = composite(model, val_imgs, val_y, dev)
    print(f"[start = original checkpoint] val_sel composite={base_comp:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in base_aucs.items()), flush=True)

    dl = DataLoader(DS(sp["train"], TrainTF(cfg)), batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                    pin_memory=True, persistent_workers=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    log = {"variant": variant, "cfg": cfg, "start": {"composite": base_comp, "aucs": base_aucs}, "epochs": []}
    best = -1.0
    out_pt = os.path.join("models", f"candidate_{variant}.pt")
    for ep in range(EPOCHS):
        model.train()
        t0, tot = time.time(), 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1).float(), y * 0.98 + 0.01)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if i % 100 == 0:
                print(f"  ep{ep + 1} it{i}/{len(dl)} loss={loss.item():.4f}", flush=True)
        comp, aucs = composite(model, val_imgs, val_y, dev)
        print(f"epoch {ep + 1}/{EPOCHS} loss={tot / len(dl):.4f} val_sel composite={comp:.4f} " +
              " ".join(f"{k}={v:.3f}" for k, v in aucs.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        log["epochs"].append({"epoch": ep + 1, "loss": tot / len(dl), "composite": comp, "aucs": aucs})
        if comp > best:
            best = comp
            torch.save({"arch": ck["arch"], "img": ck["img"], "state": model.state_dict(), "epoch": ep + 1,
                        "val_sel_composite": comp, "variant": variant, "init": os.path.basename(INIT)}, out_pt)
            log["best_epoch"] = ep + 1
            log["best_composite"] = comp
    json.dump(log, open(os.path.join("artifacts", f"train_{variant}.json"), "w"), indent=1)
    print(f"done. best epoch={log['best_epoch']} composite={best:.4f} (original: {base_comp:.4f}) -> {out_pt}")


if __name__ == "__main__":
    main(sys.argv[1])
