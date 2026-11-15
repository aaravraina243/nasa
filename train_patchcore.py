r"""
Train a supervised urine-sediment classifier (replaces the one-class PatchCore model).

Dataset: Roboflow "folder" export  ->  <root>/train, <root>/valid, <root>/test, and inside each
split one sub-folder per class, e.g.  "cast", "cryst epithn leuko", "eryth mycete" ...
Each folder name is a COMBINATION of up to 6 findings, so this is trained as MULTI-LABEL:

    cast    = casts
    cryst   = crystals            <- the "kidney stone" signal used by the web app
    epith   = epithelial cells    (folder spellings "epith" and "epithn" are treated the same)
    eryth   = red blood cells
    leuko   = white blood cells
    mycete  = yeast / fungi

Folders with other names (e.g. "JPEGImages") are skipped automatically.

How to use (VS Code)
--------------------
  1. Edit the SETTINGS section below (ZIP_PATH and OUT_DIR at minimum).
  2. Press the "Run Python File" (play) button.

Output (in OUT_DIR):  patchcore_model.pt  (same file name as before, so run_patchcore.py,
patchcore_service.py and server.js keep working), metrics.json, scores.csv.
Needs internet on the first run (ImageNet backbone weights).
"""
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import laplace
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =====================================================================
#  SETTINGS  -  EDIT THIS SECTION, THEN PRESS "RUN" IN VS CODE
# =====================================================================
ZIP_PATH = r"C:\Users\aarav\OneDrive\nasa_space_app\Urine.v1i.folder.zip"
DATA_DIR = None             # if you already unzipped it, put the folder that holds train/valid/test here
OUT_DIR = r"C:\Users\aarav\OneDrive\nasa_space_app"   # model is saved here (and the zip is unpacked to OUT_DIR\urine_dataset)

BACKBONE = "resnet18"       # resnet18 | resnet34 | resnet50 | efficientnet_b0  (resnet18 = fastest on CPU)
IMG_SIZE = 320              # images are resized to IMG_SIZE x IMG_SIZE
EPOCHS = 15
BATCH_SIZE = 32             # lower to 16 if you run out of memory
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2             # set 0 if Windows complains about data-loader workers
SEED = 42
# =====================================================================

LABELS = ["cast", "cryst", "epith", "eryth", "leuko", "mycete"]
TOKEN_MAP = {"cast": "cast", "cryst": "cryst", "epith": "epith", "epithn": "epith",
             "eryth": "eryth", "leuko": "leuko", "mycete": "mycete"}
STONE_LABEL = "cryst"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ helpers (also imported by run_patchcore.py)
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)


def load_pil(path):
    return Image.open(path).convert("RGB")


def focus_score(path):
    """Variance of the Laplacian: low value = blurry / out of focus."""
    g = np.asarray(load_pil(path).convert("L"), dtype=np.float32)
    return float(laplace(g).var())


def build_transform(size_hw):
    """Inference transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(tuple(size_hw), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class Rot90:
    """Random multiple-of-90-degree rotation (microscopy has no 'up')."""
    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 3))


def build_train_transform(size_hw):
    return transforms.Compose([
        transforms.Resize(tuple(size_hw), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Rot90(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),   # mild: colour is part of the signal
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(arch, n_out, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if arch in ("resnet18", "resnet34", "resnet50"):
        net = getattr(models, arch)(weights=w)
        net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.fc.in_features, n_out))
    elif arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=w)
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.classifier[1].in_features, n_out))
    else:
        raise ValueError("BACKBONE must be resnet18, resnet34, resnet50 or efficientnet_b0")
    return net


# ------------------------------------------------------------------ data
def find_dataset_root(base):
    base = Path(base)
    for cand in [base] + sorted(p for p in base.rglob("*") if p.is_dir()):
        if (cand / "train").is_dir() and ((cand / "valid").is_dir() or (cand / "val").is_dir()):
            return cand
    raise SystemExit(f"Could not find train/valid folders inside {base}")


def prepare_dataset(zip_path, data_dir, out_dir):
    if data_dir:
        return find_dataset_root(data_dir)
    target = Path(out_dir) / "urine_dataset"
    if not target.exists() or not any(target.iterdir()):
        if not Path(zip_path).is_file():
            raise SystemExit(f"ZIP_PATH not found: {zip_path}\nEdit the SETTINGS section at the top of this file.")
        print(f"Unzipping {zip_path} -> {target} (one time only) ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
    return find_dataset_root(target)


def folder_to_vector(name):
    toks = name.lower().split()
    if not toks or any(t not in TOKEN_MAP for t in toks):
        return None
    v = np.zeros(len(LABELS), dtype=np.float32)
    for t in toks:
        v[LABELS.index(TOKEN_MAP[t])] = 1.0
    return v


def collect_split(root, names):
    split_dir = next((root / n for n in names if (root / n).is_dir()), None)
    if split_dir is None:
        return []
    items, skipped = [], []
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        v = folder_to_vector(d.name)
        if v is None:
            skipped.append(d.name)
            continue
        items += [(p, v) for p in list_images(d)]
    if skipped:
        print(f"  [{split_dir.name}] skipped folders: {', '.join(skipped)}")
    return items


class UrineDS(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tfm(load_pil(p)), torch.from_numpy(y)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device)).float()).cpu())
        ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def macro_auc(Y, P):
    a = [safe_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
    a = [x for x in a if x is not None]
    return float(np.mean(a)) if a else 0.0


def best_threshold(y, p):
    """Threshold that maximises F1 on the validation split."""
    if y.sum() == 0:
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float(((~pred) & (y == 1)).sum())
        f = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def label_report(Y, P, thr):
    rep = {}
    for j, lab in enumerate(LABELS):
        y, p = Y[:, j], P[:, j]
        pred = p >= thr[lab]
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
        rep[lab] = {
            "n_positive": int(y.sum()), "n_total": int(len(y)), "threshold": thr[lab],
            "auroc": safe_auc(y, p),
            "avg_precision": float(average_precision_score(y, p)) if y.sum() > 0 else None,
            "precision": tp / max(tp + fp, 1), "recall (sensitivity)": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        }
    rep["exact_match_accuracy"] = float(np.mean([(P[i] >= [thr[l] for l in LABELS]).astype(int).tolist()
                                                 == Y[i].astype(int).tolist() for i in range(len(Y))]))
    return rep


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip_path", default=ZIP_PATH)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--backbone", default=BACKBONE, choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    a, _ = ap.parse_known_args()

    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    root = prepare_dataset(a.zip_path, a.data_dir, a.out_dir)
    print(f"Dataset root: {root}")
    train_items = collect_split(root, ["train"])
    val_items = collect_split(root, ["valid", "val"])
    test_items = collect_split(root, ["test"])
    if not train_items or not val_items:
        raise SystemExit("Need images in both the train and valid splits.")
    print(f"device={device} | train={len(train_items)} | valid={len(val_items)} | test={len(test_items)}")
    Ytr = np.stack([y for _, y in train_items])
    for j, lab in enumerate(LABELS):
        print(f"  {lab:7s} positives in train: {int(Ytr[:, j].sum())}")
    if device.type == "cpu":
        print("[note] running on CPU: expect several minutes per epoch. Lower IMG_SIZE or EPOCHS to go faster.")
    print("[note] Roboflow exports can contain augmented copies of the same photo in different splits; "
          "if test scores look too good, that is the likely reason.")

    size_hw = (a.img_size, a.img_size)
    mk = lambda items, tfm, shuffle: DataLoader(UrineDS(items, tfm), batch_size=a.batch_size, shuffle=shuffle,
                                                num_workers=a.num_workers, pin_memory=device.type == "cuda",
                                                drop_last=False)
    train_dl = mk(train_items, build_train_transform(size_hw), True)
    val_dl = mk(val_items, build_transform(size_hw), False)
    test_dl = mk(test_items, build_transform(size_hw), False) if test_items else None

    model = build_model(a.backbone, len(LABELS), pretrained=True).to(device)
    pos = Ytr.sum(0)
    pos_weight = torch.tensor(np.clip((len(Ytr) - pos) / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train_dl))
    use_amp = device.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))

    best_auc, best_state = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for bi, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
            print(f"  epoch {ep}/{a.epochs}  batch {bi}/{len(train_dl)}", end="\r")
        P, Y = predict(model, val_dl, device)
        auc = macro_auc(Y, P)
        stone_auc = safe_auc(Y[:, LABELS.index(STONE_LABEL)], P[:, LABELS.index(STONE_LABEL)])
        mark = ""
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best"
        print(f"epoch {ep}/{a.epochs}  train_loss={tot / n:.4f}  valid macro-AUROC={auc:.4f}  "
              f"{STONE_LABEL}-AUROC={stone_auc if stone_auc is None else round(stone_auc, 4)}{mark}")

    model.load_state_dict(best_state)
    P_val, Y_val = predict(model, val_dl, device)
    thr = {lab: best_threshold(Y_val[:, j], P_val[:, j]) for j, lab in enumerate(LABELS)}
    print("Thresholds tuned on the validation split:", {k: round(v, 2) for k, v in thr.items()})

    print("Computing focus statistics ...")
    sample = random.Random(a.seed).sample(train_items, min(300, len(train_items)))
    focus = [focus_score(p) for p, _ in sample]
    focus_min = float(np.percentile(focus, 1) * 0.5)

    metrics = {"backbone": a.backbone, "img_size": a.img_size, "best_valid_macro_auroc": best_auc,
               "valid": label_report(Y_val, P_val, thr)}
    rows = [(str(p), "valid", *[f"{x:.4f}" for x in pr]) for (p, _), pr in zip(val_items, P_val)]
    if test_dl is not None:
        P_te, Y_te = predict(model, test_dl, device)
        metrics["test"] = label_report(Y_te, P_te, thr)
        metrics["test_macro_auroc"] = macro_auc(Y_te, P_te)
        rows += [(str(p), "test", *[f"{x:.4f}" for x in pr]) for (p, _), pr in zip(test_items, P_te)]
        print(f"TEST macro-AUROC = {metrics['test_macro_auroc']:.4f}")
        for lab in LABELS:
            r = metrics["test"][lab]
            print(f"  {lab:7s} AUROC={r['auroc'] if r['auroc'] is None else round(r['auroc'], 3)}  "
                  f"recall={r['recall (sensitivity)']:.3f}  precision={r['precision']:.3f}  (n+={r['n_positive']})")

    torch.save({
        "state_dict": best_state, "arch": a.backbone, "labels": LABELS, "thresholds": thr,
        "stone_label": STONE_LABEL, "threshold": thr[STONE_LABEL], "size_hw": size_hw,
        "focus_min": focus_min, "val_scores": P_val[:, LABELS.index(STONE_LABEL)].tolist(),
    }, out_dir / "patchcore_model.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "split"] + [f"p_{l}" for l in LABELS]); w.writerows(rows)
    print(f"\nDone. Model saved to: {out_dir / 'patchcore_model.pt'}")


if __name__ == "__main__":
    main()
