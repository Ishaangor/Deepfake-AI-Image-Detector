"""End-to-end fine-tuning of a pretrained CNN for real-vs-fake face detection (PyTorch, GPU).

Label convention: fake = 0, real = 1 (same as the Keras pipeline). Output = P(real).
Saves models/best_finetuned.pt and models/calibration.json (temperature scaling).
"""
import glob
import io
import json
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

SEED = 42
ARCH = os.environ.get("ARCH", "efficientnet_b0")
IMG = int(os.environ.get("IMG", 256))
EPOCHS = int(os.environ.get("EPOCHS", 8))
BATCH = int(os.environ.get("BATCH", 32))
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", 0))  # 0 = all (use a small value for a smoke test)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

D = "data"
AI = os.path.join(D, "ai_faces", "AI-face-detection-Dataset")


def imgs(p):
    return sorted(glob.glob(os.path.join(p, "*.*")))


# ---- collect data -----------------------------------------------------------
ai_fake, ai_real = imgs(os.path.join(AI, "AI")), imgs(os.path.join(AI, "real"))
rng = random.Random(SEED)
rng.shuffle(ai_fake)
rng.shuffle(ai_real)


def split_ai(files):
    n = len(files)
    return files[: int(.8 * n)], files[int(.8 * n): int(.9 * n)], files[int(.9 * n):]


af_tr, af_va, af_te = split_ai(ai_fake)
ar_tr, ar_va, ar_te = split_ai(ai_real)


def dfd(split, cls):
    return imgs(os.path.join(D, "deepfake_dataset", split, cls))


train = [(p, 0) for p in dfd("train", "fake") + af_tr] + [(p, 1) for p in dfd("train", "real") + ar_tr]
val = [(p, 0) for p in dfd("validation", "fake") + af_va] + [(p, 1) for p in dfd("validation", "real") + ar_va]
test = [(p, 0) for p in dfd("test", "fake") + af_te] + [(p, 1) for p in dfd("test", "real") + ar_te]
ext = [(p, 0) for p in imgs(os.path.join(D, "rvf10k", "valid", "fake"))] + \
      [(p, 1) for p in imgs(os.path.join(D, "rvf10k", "valid", "real"))]


def cap(items):
    if not MAX_PER_CLASS:
        return items
    out = []
    for c in (0, 1):
        out += [x for x in items if x[1] == c][:MAX_PER_CLASS]
    return out


train, val, test, ext = map(cap, (train, val, test, ext))
print(f"train={len(train)} val={len(val)} test={len(test)} external={len(ext)}  arch={ARCH} img={IMG} dev={dev}")

# ---- transforms (heavy aug so it generalises to unseen generators / social-media JPEGs) ----
norm = T.Normalize(mean=timm.data.IMAGENET_DEFAULT_MEAN, std=timm.data.IMAGENET_DEFAULT_STD)


class RandJPEG:
    def __call__(self, im):
        if random.random() < 0.4:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=random.randint(50, 95))
            b.seek(0)
            im = Image.open(b).convert("RGB")
        return im


tf_train = T.Compose([
    RandJPEG(),
    T.RandomResizedCrop(IMG, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.2),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), norm,
])
tf_eval = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), torch.tensor(y, dtype=torch.float32)


def loader(items, tf, shuffle):
    nw = int(os.environ.get("WORKERS", 5)) if shuffle else 0
    return DataLoader(DS(items, tf), batch_size=BATCH, shuffle=shuffle, num_workers=nw,
                      pin_memory=True, persistent_workers=shuffle, drop_last=shuffle)


def main():
    dl_tr, dl_va, dl_te, dl_ex = loader(train, tf_train, True), loader(val, tf_eval, False), \
        loader(test, tf_eval, False), loader(ext, tf_eval, False)

    model = timm.create_model(ARCH, pretrained=True, num_classes=1, drop_rate=0.3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=EPOCHS * len(dl_tr), pct_start=0.15)
    scaler = torch.amp.GradScaler()

    @torch.no_grad()
    def logits_of(dl):
        model.eval()
        zs, ys = [], []
        for x, y in dl:
            with torch.autocast(dev, dtype=torch.float16):
                z = model(x.to(dev, non_blocking=True)).squeeze(1)
            zs.append(z.float().cpu())
            ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    def report(name, z, y, temp=1.0):
        p = torch.sigmoid(z / temp).numpy()
        yy = y.numpy()
        conf = np.where(p >= .5, p, 1 - p)
        print(f"  {name}: acc={accuracy_score(yy, p >= .5) * 100:.2f}% auc={roc_auc_score(yy, p):.4f} "
              f"mean_conf={conf.mean() * 100:.1f}% frac>=90%={np.mean(conf >= .9) * 100:.1f}%", flush=True)

    best = -1
    os.makedirs("models", exist_ok=True)
    for ep in range(EPOCHS):
        model.train()
        tot = 0
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            ys = y * 0.98 + 0.01  # light label smoothing
            with torch.autocast(dev, dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1).float(), ys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep + 1} it{i}/{len(dl_tr)} loss={loss.item():.4f}", flush=True)
        zv, yv = logits_of(dl_va)
        auc = roc_auc_score(yv.numpy(), zv.numpy())
        print(f"epoch {ep + 1}/{EPOCHS} train_loss={tot / len(dl_tr):.4f}")
        report("val", zv, yv)
        if auc > best:
            best = auc
            torch.save({"arch": ARCH, "img": IMG, "state": model.state_dict()}, "models/best_finetuned.pt")

    # ---- reload best, calibrate temperature on val, evaluate ----
    ck = torch.load("models/best_finetuned.pt", map_location=dev)
    model.load_state_dict(ck["state"])
    zv, yv = logits_of(dl_va)
    temp = nn.Parameter(torch.ones(1))
    o = torch.optim.LBFGS([temp], lr=0.1, max_iter=100)

    def closure():
        o.zero_grad()
        l = F.binary_cross_entropy_with_logits(zv / temp, yv)
        l.backward()
        return l

    o.step(closure)
    tc = float(temp.item())
    print(f"\nTemperature = {tc:.3f}")
    json.dump({"temperature": tc, "arch": ARCH, "img": IMG}, open("models/calibration.json", "w"))

    for name, dl in (("TEST (held out)", dl_te), ("EXTERNAL rvf10k/valid", dl_ex)):
        z, y = logits_of(dl)
        print(name)
        report("raw       ", z, y)
        report("calibrated", z, y, tc)


if __name__ == "__main__":
    main()
