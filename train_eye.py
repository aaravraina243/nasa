r"""
Train a supervised BINARY classifier that spots a SWOLLEN OPTIC NERVE (papilledema) in retinal (fundus) photos.
(Replaces the old one-class PatchCore model. File names, the model path and the JSON the web app gets are unchanged.)

Keep this file in the iter-10 folder (next to server.js). run_eye.py and eye_service.py import from it.

Data (two sources, both sit in the folder ABOVE iter-10; either one alone also works)
    1) Papilledema_detection.v2i.multiclass.zip -> train/_classes.csv + train/*.jpg      (Roboflow export)
       4,107 AUGMENTED images = 1,369 original photos x 3 copies (random flip / rotation / brightness / blur / noise).
       _classes.csv: filename, Normal, Papilledema, Pseudopapilledema   (one-hot 0/1)
    2) eye.zip -> Normal (779) / Papilledema (295) / Pseudopapilledema (295)             (the CLEAN original photos)
    The numbers match, so these are very likely the SAME 1,369 photos. The script links every clean photo to its
    3 augmented copies (same class + same file number, e.g. 238.jpg <-> 238_jpg.rf.xxxx.jpg; if the names do not
    match, that clean photo is left out to be safe) and keeps a photo + all its copies together in ONE split.
    Otherwise the test scores would look far better than they really are.
    TRAINING uses all of them (clean + augmented). EVALUATION (early stopping, threshold, test set) uses only the
    CLEAN originals, i.e. photos like the ones the web app will receive.

The task is binary:   1 = Papilledema        0 = not papilledema
What happens to "Pseudopapilledema" (harmless look-alike, e.g. optic disc drusen) is set by PSEUDO_MODE below:
    "negative" (default) -> it counts as "not papilledema". The model must learn to tell the look-alike apart,
                            which is what you want clinically. Results are reported separately for
                            Normal and for Pseudopapilledema so you can see how each one behaves.
    "exclude"            -> ignored completely (what the old script did)
    "positive"           -> counts as "abnormal disc" together with Papilledema

How it learns (built for a SMALL data set: ~1,400 images)
    * A photo, its clean original and its 3 augmented copies form one group, and a group is never split across
      train / validation / test, so the scores are not inflated by leakage.
    * 15% of the images are held out as a TEST set that is never used for any decision.
    * The rest is trained with 5-fold cross-validation. Each fold fine-tunes an ImageNet CNN with strong
      augmentation, class-balanced sampling, label smoothing, EMA weights and early stopping.
    * The 5 fold-models are kept as an ENSEMBLE (+ flip test-time augmentation) = more accurate and more stable.
    * The out-of-fold predictions (every training image scored by a model that never saw it) choose the decision
      threshold (strictest one that still catches MIN_SENSITIVITY of papilledema) and calibrate the 0-1
      probability (exactly 0.5 at the threshold).
    * The TEST set then gives the fair performance estimate, with 95% confidence intervals (resampled per original
      photo, not per copy), plus a check of how
      the model copes with blurry / dark / bright / low-contrast photos.
    * Results go to the models folder:  eye_model.pt , eye_metrics.json , eye_scores.csv

Run it
    cd C:\Users\aarav\OneDrive\nasa_space_app\iter-10
    python train_eye.py
(or press "Run Python File" in VS Code). The first run downloads the ImageNet weights (needs internet).
"""
import argparse
import copy
import csv
import json
import math
import random
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent

# =====================================================================
#  SETTINGS  -  EDIT IF YOUR FOLDERS ARE DIFFERENT
# =====================================================================
ROBOFLOW_PATH = str(HERE.parent / "Papilledema_detection.v2i.multiclass.zip")   # augmented export (.zip or extracted folder with _classes.csv); None = skip
ORIGINALS_PATH = str(HERE.parent / "eye.zip")                                    # clean originals (.zip or folder with Normal/Papilledema/Pseudopapilledema); None = skip
OUT_DIR = str(HERE / "models")                 # eye_model.pt, eye_metrics.json, eye_scores.csv are written here

PSEUDO_MODE = "negative"                       # "negative" | "exclude" | "positive"   (see the top of this file)

# --- model ---
BACKBONE = "auto"           # auto = convnext_tiny on GPU, resnet18 on CPU. Or: resnet18|resnet34|resnet50|
                            #        efficientnet_b0|efficientnet_v2_s|convnext_tiny
SIZE = 320                  # images are resized to SIZE x SIZE (384 = more detail, slower)

# --- training ---
FOLDS = 5                   # cross-validation folds = number of models in the ensemble (3 = faster)
EPOCHS = 15                 # per fold; one epoch already shows every photo 3 times (3 augmented copies); early stopping may end it sooner
PATIENCE = 5                # stop a fold if no improvement for this many epochs
BATCH_SIZE = 16             # lower to 8 if you run out of GPU memory
LR = 1e-4                   # backbone learning rate
HEAD_LR_MULT = 10.0         # the new output layer learns this many times faster
WEIGHT_DECAY = 1e-2
LABEL_SMOOTH = 0.05
EMA_DECAY = 0.995
BALANCED_SAMPLING = True    # each epoch shows Normal / Papilledema / Pseudopapilledema equally often
NUM_WORKERS = 2             # set 0 if Windows complains about data-loader workers
SEED = 0

# --- decision rule / evaluation ---
MIN_SENSITIVITY = 0.90      # the threshold is the strictest one that still catches this share of papilledema
TEST_FRACTION = 0.15
UNLINKED_ORIGINALS = "drop"  # clean photos whose class+number matches no augmented copy: "drop" (safe) or "keep" (if eye.zip holds different photos)
USE_TTA = True              # average predictions over flipped copies (also used by run_eye.py)
ROBUSTNESS_CHECK = True
# =====================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASS_FOLDERS = ["normal", "papilledema", "pseudopapilledema"]      # folder names (case-insensitive), index = class id
CLASS_TITLES = ["Normal", "Papilledema", "Pseudopapilledema"]
ARCHS = ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny"]


# ------------------------------------------------------------------ images (same helpers as before)
def list_images(p):
    """Image files in a file path or a folder (recursive, sorted)."""
    p = Path(p)
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []
    if p.is_dir():
        return sorted(f for f in p.rglob("*")
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                      and not f.name.startswith("._") and "__MACOSX" not in f.parts)
    return []


def load_pil(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def build_transform(size_hw, extra=None):
    """Inference transform. `extra` = optional PIL operation (used only by the robustness check)."""
    ops = [transforms.Resize(tuple(size_hw))]
    if extra is not None:
        ops.append(extra)
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def build_train_transform(size_hw):
    # Fundus photos have no fixed "up", so flips/rotations are safe. Crops stay large so the optic disc
    # is rarely cut out. Colour changes are mild (disc colour and haemorrhages are part of the signal).
    return transforms.Compose([
        transforms.RandomResizedCrop(tuple(size_hw), scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),   # focus varies in real life
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def focus_score(path):
    """Sharpness = variance of the Laplacian on a fixed-size grey image (higher = sharper)."""
    g = load_pil(path).convert("L").resize((256, 256))
    return float(laplace(np.asarray(g, dtype=np.float32)).var())


# ------------------------------------------------------------------ model (also imported by run_eye.py)
def build_model(arch, n_out=1, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if arch in ("resnet18", "resnet34", "resnet50"):
        net = getattr(models, arch)(weights=w)
        net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, n_out))
    elif arch in ("efficientnet_b0", "efficientnet_v2_s"):
        net = getattr(models, arch)(weights=w)
        net.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.classifier[1].in_features, n_out))
    elif arch == "convnext_tiny":
        net = models.convnext_tiny(weights=w)
        net.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.classifier[2].in_features, n_out))
    else:
        raise ValueError(f"BACKBONE must be one of {ARCHS}")
    return net


def head_parameters(model):
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    return list(head.parameters())


@torch.no_grad()
def ensemble_logit(models_, x, tta=False):
    """Image score = mean logit over all models (and, with tta, over flipped copies). Shape (B,)."""
    views = [x] + ([x.flip(3), x.flip(2), x.flip((2, 3))] if tta else [])
    return torch.stack([m(v).float().squeeze(1) for m in models_ for v in views]).mean(0)


def probability(score, thr, w, scale):
    """0-1 risk indicator, exactly 0.5 at the threshold (same formula as before)."""
    return 1.0 / (1.0 + math.exp(-max(min(w * (float(score) - thr) / scale, 50.0), -50.0)))


class EMA:
    """Moving average of the weights: smoother and usually more accurate than the raw weights."""
    def __init__(self, model, decay):
        self.m = copy.deepcopy(model).eval()
        self.decay, self.step = decay, 0
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


# ------------------------------------------------------------------ data (Roboflow multi-class export)
def prepare_data(data, tmp, sub):
    data = Path(data)
    if data.is_file() and data.suffix.lower() == ".zip":
        print(f"Extracting {data.name} to a temporary folder (this takes a moment) ...", flush=True)
        with zipfile.ZipFile(data) as z:
            z.extractall(Path(tmp) / sub)
        return Path(tmp) / sub
    if data.is_dir():
        return data
    raise SystemExit(f"Data not found: {data}\nEdit ROBOFLOW_PATH / ORIGINALS_PATH at the top of train_eye.py (the .zip file or its extracted folder).")


def norm_stem(name):
    """'238_jpg.rf.<hash>.jpg' (Roboflow copy) and '238.jpg' (clean original) both give '238'."""
    stem = name.split(".rf.")[0] if ".rf." in name else Path(name).stem
    return re.sub(r"_(jpe?g|png|bmp|tiff?|webp)$", "", stem.strip().lower())


def find_class_dir(root, name):
    hits = [d for d in Path(root).rglob("*") if d.is_dir() and d.name.lower() == name and "__MACOSX" not in d.parts]
    hits.sort(key=lambda d: len(list_images(d)), reverse=True)
    return hits[0] if hits else None


def read_class_folders(root):
    """Clean originals from eye.zip: rows of (image_path, class_id, group_key)."""
    rows = []
    for ci, folder in enumerate(CLASS_FOLDERS):
        d = find_class_dir(root, folder)
        if d is None:
            print(f"[warn] no '{CLASS_TITLES[ci]}' folder found in the originals" + (" (ignored)" if ci == 2 else ""))
            continue
        imgs = list_images(d)
        rows += [(p, ci, f"{ci}|{norm_stem(p.name)}") for p in imgs]
        print(f"originals / {CLASS_TITLES[ci]:18s}: {len(imgs)} images ({d})")
    return rows


def find_csvs(root):
    return sorted(c for c in Path(root).rglob("_classes.csv") if "__MACOSX" not in c.parts)


def read_classes_csv(csv_path):
    """Returns (rows, n_bad, n_missing). rows = (image_path, class_id, group_key); class_id 0/1/2 = Normal/Papilledema/Pseudopapilledema.
    group_key = class + file number  ->  all augmented copies of one photo (and its clean original) share it.
    (The class is part of the key because the numbering restarts in every class: e.g. '238_jpg' exists in several classes.)"""
    csv_path = Path(csv_path)
    rows, bad, missing = [], 0, 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {(h or "").strip().lower(): h for h in (rd.fieldnames or [])}
        fcol, ccols = cols.get("filename"), [cols.get(n) for n in CLASS_FOLDERS]
        if fcol is None or any(c is None for c in ccols):
            raise SystemExit(f"{csv_path} needs the columns: filename, Normal, Papilledema, Pseudopapilledema (found: {list(cols)})")
        for r in rd:
            try:
                flags = [int(float(r[c] or 0)) for c in ccols]
            except ValueError:
                bad += 1
                continue
            if sum(flags) != 1:
                bad += 1
                continue
            ci, name = flags.index(1), r[fcol].strip()
            p = csv_path.parent / name
            if not p.is_file():
                missing += 1
                continue
            rows.append((p, ci, f"{ci}|{norm_stem(name)}"))
    return rows, bad, missing


def link_check(paths, keys, is_orig):
    """Keep-mask. A clean original is only safe to use if it is linked (same class + file number) to the
    augmented copies of the same photo; otherwise it could sit in the test set while its copies are in training."""
    keys, is_orig = np.array(keys, dtype=object), np.asarray(is_orig, bool)
    keep = np.ones(len(keys), bool)
    if not (is_orig.any() and (~is_orig).any()):
        return keep
    aug_keys = set(keys[~is_orig].tolist())
    linked = np.array([k in aug_keys for k in keys])
    n_orig, n_linked = int(is_orig.sum()), int((is_orig & linked).sum())
    print(f"{n_linked} of {n_orig} clean photos were linked to their augmented copies by file name (class + number)")
    unl = np.where(is_orig & ~linked)[0]
    if len(unl):
        ex = ", ".join(Path(paths[i]).name for i in unl[:5])
        if UNLINKED_ORIGINALS == "drop":
            keep[unl] = False
            print(f"  {len(unl)} clean photos match no augmented copy (e.g. {ex}) -> LEFT OUT, because they could be the same "
                  "photos under another name and would then leak into the test set.")
        else:
            print(f"  {len(unl)} clean photos match no augmented copy (e.g. {ex}) -> kept as separate photos (UNLINKED_ORIGINALS = 'keep').")
        if n_linked < 0.5 * n_orig:
            print("  [warn] most clean photos could not be linked. Either eye.zip holds DIFFERENT photos (then set "
                  "UNLINKED_ORIGINALS = 'keep') or the file names differ (then send me a few names from both sets).")
    return keep


def assign_groups(keys, is_orig):
    """Returns (group id per image, is_eval per image).
    One group = one original photo: its clean original (eye.zip) + its augmented copies (Roboflow).
    is_eval = images used for evaluation: the clean originals, or the augmented copies if a photo has no clean original."""
    is_orig = np.asarray(is_orig, bool)
    _, groups = np.unique(list(keys), return_inverse=True)
    orig_in_group = np.zeros(groups.max() + 1, bool)
    orig_in_group[groups[is_orig]] = True
    return groups, is_orig | ~orig_in_group[groups]


class EyeDS(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm        # items: (path, binary_label, class_id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, _ = self.items[i]
        return self.tfm(load_pil(p)), torch.tensor(float(y))


# ------------------------------------------------------------------ metrics
def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def choose_threshold(s, y, min_sens):
    """Strictest threshold (flag when score >= thr) that still catches `min_sens` of the positives."""
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos = s[y == 1]
    cand = np.unique(s)
    best_i = 0
    for i, t in enumerate(cand):
        if (pos >= t).mean() >= min_sens:
            best_i = i
    t = cand[best_i]
    lower = cand[best_i - 1] if best_i > 0 else t - 1.0
    return float((t + lower) / 2)          # midway, so the boundary image is not decided by rounding


def fit_slope(s, y, thr, scale):
    """Steepness w of p = sigmoid(w * (score - thr) / scale), fitted by log-loss. p is exactly 0.5 at thr."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    z = (s - thr) / scale
    best_w, best_ll = 1.0, None
    for w in np.linspace(0.2, 8.0, 157):
        p = np.clip(1 / (1 + np.exp(-np.clip(w * z, -50, 50))), 1e-6, 1 - 1e-6)
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if best_ll is None or ll < best_ll:
            best_w, best_ll = float(w), ll
    return best_w


def bootstrap_ci(s, y, thr, groups=None, n=1000, seed=0):
    """95% intervals, resampling whole ORIGINAL PHOTOS (a photo and its augmented copies move together)."""
    rng = np.random.default_rng(seed)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    by = {c: [np.where(groups == g)[0] for g in np.unique(groups[y == c])] for c in (0, 1)}
    aucs, sens, spec = [], [], []
    for _ in range(n):
        ip = np.concatenate([by[1][i] for i in rng.integers(0, len(by[1]), len(by[1]))])
        ineg = np.concatenate([by[0][i] for i in rng.integers(0, len(by[0]), len(by[0]))])
        idx = np.concatenate([ip, ineg])
        aucs.append(roc_auc_score(y[idx], s[idx]))
        sens.append((s[ip] >= thr).mean())
        spec.append((s[ineg] < thr).mean())
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"auroc_95ci": ci(aucs), "sensitivity_95ci": ci(sens), "specificity_95ci": ci(spec)}


def report(s, y, cls, thr, w, scale, with_ci=False, groups=None):
    s, y, cls = np.asarray(s, float), np.asarray(y, int), np.asarray(cls, int)
    pred = s >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else None
    probs = np.array([probability(v, thr, w, scale) for v in s])
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    rep = {"n_images": int(len(y)), "n_papilledema": int((y == 1).sum()), "n_not_papilledema": int((y == 0).sum()),
           "auroc": auroc(y, s), "avg_precision": float(average_precision_score(y, s)) if y.sum() else None,
           "sensitivity": sens, "specificity": spec, "precision": div(tp, tp + fp),
           "negative_predictive_value": div(tn, tn + fn), "accuracy": div(tp + tn, len(y)),
           "balanced_accuracy": (sens + spec) / 2 if sens is not None and spec is not None else None,
           "confusion": {"true_negative": tn, "false_alarm": fp, "missed_papilledema": fn, "true_papilledema": tp},
           "brier_score": float(((probs - y) ** 2).mean()),
           "share_flagged_by_class": {CLASS_TITLES[c]: {"n": int((cls == c).sum()), "flagged_as_papilledema": float(pred[cls == c].mean())}
                                      for c in range(3) if (cls == c).any()}}
    if with_ci and (y == 1).any() and (y == 0).any():
        rep.update(bootstrap_ci(s, y, thr, groups))
    return rep


# ------------------------------------------------------------------ training
@torch.no_grad()
def predict_scores(models_, loader, device, tta):
    for m in models_:
        m.eval()
    ss, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            ss.append(ensemble_logit(models_, x.to(device), tta).cpu())
        ys.append(y)
    return torch.cat(ss).numpy(), torch.cat(ys).numpy().astype(int)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_fold(fold, arch, train_items, val_items, size_hw, device, a):
    seed_everything(a.seed + fold)
    pin, persistent = device.type == "cuda", a.num_workers > 0
    labels = np.array([y for _, y, _ in train_items], dtype=np.float32)
    sampler, pos_weight = None, float((labels == 0).sum() / max((labels == 1).sum(), 1))
    if BALANCED_SAMPLING:
        cls = np.array([c for _, _, c in train_items])
        cnt = np.bincount(cls, minlength=3)
        wts = np.array([1.0 / cnt[c] for c in cls])
        sampler = WeightedRandomSampler(torch.from_numpy(wts), num_samples=len(wts), replacement=True)
        pos_weight = 1.0                                        # sampling already balances the classes
    train_dl = DataLoader(EyeDS(train_items, build_train_transform(size_hw)), batch_size=a.batch_size,
                          sampler=sampler, shuffle=sampler is None, drop_last=len(train_items) > a.batch_size,
                          num_workers=a.num_workers, pin_memory=pin, persistent_workers=persistent)
    val_dl = DataLoader(EyeDS(val_items, build_transform(size_hw)), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=pin, persistent_workers=persistent)

    model = build_model(arch, 1, pretrained=not a.no_pretrained).to(device)
    ema = EMA(model, EMA_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    head_ids = {id(p) for p in head_parameters(model)}
    backbone_p = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone_p, "lr": a.lr},
                             {"params": head_parameters(model), "lr": a.lr * HEAD_LR_MULT}], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * HEAD_LR_MULT], epochs=a.epochs,
                                                steps_per_epoch=len(train_dl), pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))

    best, best_state, best_scores, best_y, bad = -1.0, None, None, None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for bi, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            ys = y * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x).float().squeeze(1), ys)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            tot += loss.item() * len(x); n += len(x)
            print(f"  fold {fold + 1} epoch {ep}/{a.epochs}  batch {bi}/{len(train_dl)}", end="\r")
        s, yv = predict_scores([ema.m], val_dl, device, USE_TTA)
        auc, ap = auroc(yv, s) or 0.0, float(average_precision_score(yv, s))
        score, mark = 0.5 * auc + 0.5 * ap, ""
        if score > best:
            best, bad, mark = score, 0, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in ema.m.state_dict().items()}
            best_scores, best_y = s, yv
        else:
            bad += 1
        print(f"fold {fold + 1} epoch {ep}/{a.epochs}  train_loss={tot / n:.4f}  valid AUROC={auc:.4f}  AP={ap:.4f}{mark}")
        if bad >= a.patience:
            print(f"fold {fold + 1}: no improvement for {a.patience} epochs -> early stop.")
            break
    return best_state, best_scores, best_y, best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roboflow", default=ROBOFLOW_PATH)
    ap.add_argument("--originals", default=ORIGINALS_PATH)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--backbone", default=BACKBONE, choices=["auto"] + ARCHS)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--min_sensitivity", type=float, default=MIN_SENSITIVITY)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no_pretrained", action="store_true", help=argparse.SUPPRESS)   # offline smoke tests only
    a = ap.parse_args()
    if PSEUDO_MODE not in ("negative", "exclude", "positive"):
        raise SystemExit('PSEUDO_MODE must be "negative", "exclude" or "positive"')
    if a.folds < 2:
        raise SystemExit("--folds must be at least 2")

    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = a.backbone if a.backbone != "auto" else ("convnext_tiny" if device.type == "cuda" else "resnet18")
    size_hw = (a.size, a.size)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tmp_kw = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
    with tempfile.TemporaryDirectory(**tmp_kw) as tmp:
        items, group_keys, is_orig = [], [], []      # (path, binary label, class id) | one key per original photo | True = clean original

        def add(rows, orig):
            for p, ci, key in rows:
                if ci == 2 and PSEUDO_MODE == "exclude":
                    continue
                y = 1 if ci == 1 or (ci == 2 and PSEUDO_MODE == "positive") else 0
                items.append((p, y, ci))
                group_keys.append(key)
                is_orig.append(orig)

        if a.roboflow and Path(a.roboflow).exists():
            root = prepare_data(a.roboflow, tmp, "roboflow")
            csvs = find_csvs(root)
            if not csvs:
                raise SystemExit(f"No _classes.csv found inside {a.roboflow}. Point ROBOFLOW_PATH at the Roboflow multi-class export.")
            for c in csvs:
                rows, bad, missing = read_classes_csv(c)
                print(f"roboflow / {c.parent.name}/_classes.csv: {len(rows)} augmented images"
                      + (f" | skipped {bad} rows without exactly one class" if bad else "")
                      + (f" | [warn] {missing} listed image files not found" if missing else ""))
                add(rows, False)
        elif a.roboflow:
            print(f"[warn] Roboflow export not found, skipped: {a.roboflow}")
        if a.originals and Path(a.originals).exists():
            add(read_class_folders(prepare_data(a.originals, tmp, "originals")), True)
        elif a.originals:
            print(f"[warn] originals not found, skipped: {a.originals}")
        if not items:
            raise SystemExit("No images found. Check ROBOFLOW_PATH and ORIGINALS_PATH at the top of train_eye.py.")
        for ci in range(3):
            n_ci = sum(1 for _, _, c in items if c == ci)
            if n_ci:
                print(f"{CLASS_TITLES[ci]:18s}: {n_ci:5d} images in total")
        n_pos = sum(y for _, y, _ in items)
        if n_pos < 30 or len(items) - n_pos < 30:
            raise SystemExit("Need at least 30 images of each of the two classes to train and check the model.")
        print(f"device={device} | backbone={arch} | size={a.size} | folds={a.folds} | PSEUDO_MODE={PSEUDO_MODE} "
              f"| positives={n_pos} | negatives={len(items) - n_pos}")
        if device.type == "cpu":
            print("[note] running on CPU: expect long training (lower --folds / --epochs / --size to go faster).")

        # ---- group-aware, stratified splits: test set + cross-validation folds
        paths = [p for p, _, _ in items]
        keep = link_check(paths, group_keys, is_orig)
        if not keep.all():
            items = [x for x, k in zip(items, keep) if k]
            group_keys = [x for x, k in zip(group_keys, keep) if k]
            is_orig = [x for x, k in zip(is_orig, keep) if k]
            paths = [p for p, _, _ in items]
        cls_all = np.array([c for _, _, c in items])
        y_all = np.array([y for _, y, _ in items])
        is_orig = np.array(is_orig, bool)
        groups, is_eval = assign_groups(group_keys, is_orig)
        sizes = np.bincount(groups)
        from collections import Counter
        print(f"{len(items)} images = {int(is_orig.sum())} clean originals + {int((~is_orig).sum())} augmented copies, "
              f"from {len(sizes)} original photos (images per photo: {dict(Counter(sizes.tolist()))})")
        print(f"each photo and all its copies stay in ONE split | {int(is_eval.sum())} images are used for evaluation "
              f"({'clean originals' if is_orig.any() else 'augmented copies - no clean originals were loaded'})")
        if sizes.max() == 1:
            print("[warn] no augmented copies were linked to any photo, so leakage between splits cannot be ruled out.")

        idx = np.arange(len(items))
        sg = StratifiedGroupKFold(n_splits=max(2, round(1 / TEST_FRACTION)), shuffle=True, random_state=a.seed)
        dev_idx, test_idx = next(iter(sg.split(idx, cls_all, groups)))
        sg2 = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed + 1)
        folds = [(dev_idx[tr], dev_idx[va]) for tr, va in sg2.split(dev_idx, cls_all[dev_idx], groups[dev_idx])]
        dev_eval, test_eval = dev_idx[is_eval[dev_idx]], test_idx[is_eval[test_idx]]
        print(f"Test set: {len(test_eval)} evaluation images from {len(set(groups[test_idx].tolist()))} photos "
              f"(never used for any decision) | development set: {len(dev_idx)} images in {a.folds} folds")

        # ---- train one model per fold (train on everything in the training photos, validate on clean originals)
        best_states, oof = [], np.full(len(items), np.nan)
        fold_scores = []
        for k, (tr, va) in enumerate(folds):
            va_eval = va[is_eval[va]]
            print(f"\n=== fold {k + 1}/{a.folds}: train {len(tr)} | validation {len(va_eval)} ===", flush=True)
            state, s_val, y_val, sc = train_fold(k, arch, [items[i] for i in tr], [items[i] for i in va_eval],
                                                 size_hw, device, a)
            best_states.append({kk: (v.half().float() if v.is_floating_point() else v) for kk, v in state.items()})
            oof[va_eval] = s_val
            fold_scores.append(auroc(y_val, s_val))
            print(f"fold {k + 1} best validation AUROC = {fold_scores[-1]:.4f}")
        assert not np.isnan(oof[dev_eval]).any()

        # ---- decision threshold + probability calibration from out-of-fold scores
        s_dev, y_dev, c_dev = oof[dev_eval], y_all[dev_eval], cls_all[dev_eval]
        thr = choose_threshold(s_dev, y_dev, a.min_sensitivity)
        scale = float(max(s_dev.std(), 1e-6))
        w = fit_slope(s_dev, y_dev, thr, scale)
        m_oof = report(s_dev, y_dev, c_dev, thr, w, scale)

        # ---- final ensemble (exactly the weights that get saved) evaluated on the untouched test set
        print("\nScoring the test set with the ensemble ...", flush=True)
        ens = []
        for st in best_states:
            m = build_model(arch, 1, pretrained=False)
            m.load_state_dict(st)
            ens.append(m.to(device).eval())

        def make_loader(its, extra=None):
            return DataLoader(EyeDS(its, build_transform(size_hw, extra)), batch_size=a.batch_size, shuffle=False,
                              num_workers=a.num_workers, pin_memory=device.type == "cuda")

        test_items = [items[i] for i in test_eval]
        s_test, y_test = predict_scores(ens, make_loader(test_items), device, USE_TTA)
        c_test = cls_all[test_eval]
        m_test = report(s_test, y_test, c_test, thr, w, scale, with_ci=True, groups=groups[test_eval])

        # the augmented (blurred / noisy / rotated) copies of the same test photos: a harder, noisier second view
        test_aug = test_idx[~is_eval[test_idx]]
        s_ta, m_test_aug = [], None
        if len(test_aug):
            s_ta, y_ta = predict_scores(ens, make_loader([items[i] for i in test_aug]), device, USE_TTA)
            m_test_aug = report(s_ta, y_ta, cls_all[test_aug], thr, w, scale, with_ci=True, groups=groups[test_aug])

        robustness = None
        if ROBUSTNESS_CHECK:
            print("Robustness check (test set, changed photo quality) ...", flush=True)
            perturb = {"clean": None,
                       "blurred": transforms.GaussianBlur(9, sigma=(2.0, 2.0)),
                       "darker": transforms.ColorJitter(brightness=(0.7, 0.7)),
                       "brighter": transforms.ColorJitter(brightness=(1.3, 1.3)),
                       "low_contrast": transforms.ColorJitter(contrast=(0.7, 0.7))}
            robustness = {}
            for name, op in perturb.items():
                sp, yp = predict_scores(ens, make_loader(test_items, op), device, USE_TTA)
                r = report(sp, yp, c_test, thr, w, scale)
                robustness[name] = {"auroc": r["auroc"], "sensitivity": r["sensitivity"], "specificity": r["specificity"]}

        sample = random.Random(a.seed).sample(dev_eval.tolist(), min(300, len(dev_eval)))
        focus_min = float(np.percentile([focus_score(paths[i]) for i in sample], 5))

        ck = {"state_dicts": [{k: (v.half() if v.is_floating_point() else v) for k, v in st.items()} for st in best_states],
              "arch": arch, "backbone": arch, "n_models": len(best_states), "size_hw": list(size_hw),
              "threshold": float(thr), "prob_w": float(w), "prob_scale": scale, "focus_min": focus_min,
              "tta": bool(USE_TTA), "oof_scores": [float(v) for v in s_dev],
              "trained_on": "Normal vs Papilledema (supervised binary classifier)", "pseudo_mode": PSEUDO_MODE,
              "classes": ["Not papilledema", "Papilledema"], "version": 2,
              "created": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics_test": m_test}
        torch.save(ck, out / "eye_model.pt")

        metrics = {
            "created": ck["created"], "backbone": arch, "image_size": a.size, "ensemble_models": len(best_states),
            "tta": bool(USE_TTA), "pseudo_mode": PSEUDO_MODE, "threshold": float(thr),
            "probability_slope": float(w), "probability_scale": scale,
            "rule": f"flag as papilledema when the ensemble score >= threshold (chosen for >= {a.min_sensitivity:.0%} sensitivity on out-of-fold predictions)",
            "images": {"total": len(items), "clean_originals": int(is_orig.sum()), "augmented_copies": int((~is_orig).sum()),
                       "development": int(len(dev_idx)), "test": int(len(test_idx)),
                       "evaluation_images_development": int(len(dev_eval)), "evaluation_images_test": int(len(test_eval)),
                       "original_photos": int(len(sizes)), "max_copies_per_photo": int(sizes.max()),
                       "test_original_photos": int(len(set(groups[test_idx].tolist()))),
                       "by_class": {CLASS_TITLES[c]: int((cls_all == c).sum()) for c in range(3) if (cls_all == c).any()}},
            "cross_validation": {"fold_auroc": fold_scores, "mean_auroc": float(np.mean(fold_scores)),
                                 "std_auroc": float(np.std(fold_scores))},
            "test_set": m_test, "test_set_augmented_copies": m_test_aug, "out_of_fold_development_set": m_oof, "robustness_test_set": robustness,
            "notes": ["'test_set' is the fair estimate: those images (and all their augmented copies) were never used for training, early stopping, the threshold or the calibration.",
                      "'out_of_fold_development_set' is slightly optimistic because the threshold was tuned on it.",
                      "Small test sets give wide error bars - see the 95% confidence intervals.",
                      "Evaluation (validation, out-of-fold, test) uses the CLEAN original photos whenever they were loaded; augmented copies are used for training only. 'test_set_augmented_copies' shows the same test photos in their noisy/blurred/rotated versions.",
                      "All images come from one data set (probably one hospital/camera). Results on photos from a different camera or clinic can be worse; test on an external set before trusting the model.",
                      "Photos of the same patient (e.g. left and right eye) are different original photos and can land in different splits; without patient IDs this cannot be prevented.",
                      "The probability is a calibrated risk indicator for THIS data set; it has not been clinically validated."]}
        (out / "eye_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with open(out / "eye_scores.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["split", "true_class", "image", "score", "probability_papilledema", "predicted"])
            for split, ids, sc_ in (("cv_out_of_fold", dev_eval, s_dev), ("test", test_eval, s_test), ("test_augmented_copies", test_aug, s_ta)):
                for i, v in zip(ids, sc_):
                    wr.writerow([split, CLASS_TITLES[cls_all[i]], str(paths[i]), f"{v:.5f}",
                                 f"{probability(v, thr, w, scale):.4f}", "Papilledema" if v >= thr else "Not papilledema"])

    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    ci = lambda k: ("[" + ", ".join(f"{x:.3f}" for x in m_test[k]) + "]") if k in m_test else ""
    print("\n==================== RESULT (test set, not used for any tuning) ====================")
    print(f"AUROC {fmt(m_test['auroc'])} {ci('auroc_95ci')} | sensitivity {fmt(m_test['sensitivity'])} {ci('sensitivity_95ci')} "
          f"| specificity {fmt(m_test['specificity'])} {ci('specificity_95ci')}")
    print(f"confusion: {m_test['confusion']}")
    if m_test_aug:
        print(f"augmented copies of the same test photos: AUROC {fmt(m_test_aug['auroc'])} | sensitivity {fmt(m_test_aug['sensitivity'])} | specificity {fmt(m_test_aug['specificity'])}")
    for name, v in m_test["share_flagged_by_class"].items():
        print(f"  {name:18s} n={v['n']:3d}  flagged as papilledema: {v['flagged_as_papilledema']:.1%}")
    if robustness:
        for name, r in robustness.items():
            print(f"  robustness {name:13s} AUROC {fmt(r['auroc'])} | sensitivity {fmt(r['sensitivity'])} | specificity {fmt(r['specificity'])}")
    print(f"cross-validation AUROC per fold: {[round(v, 3) for v in fold_scores]}")
    print(f"threshold = {thr:.4f}   (took {(time.time() - t0) / 60:.1f} min)")
    print(f"Saved to {out}:  eye_model.pt , eye_metrics.json , eye_scores.csv")


if __name__ == "__main__":
    main()
