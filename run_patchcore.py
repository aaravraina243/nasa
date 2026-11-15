r"""
Run the trained urine-sediment classifier on new images (screening for kidney stones).

Keep this file in the SAME folder as train_patchcore.py (it reuses its code) and server.js.
It must be named run_patchcore.py, because server.js / patchcore_service.py use that name.

Two ways to use it
------------------
A) Batch mode (VS Code / "Run Python File")
     1. Edit the SETTINGS section below (MODEL_PATH, INPUT_PATH, RESULTS_DIR).
     2. Press the "Run Python File" (play) button.
   Output: predictions.csv

B) Web app mode (started by server.js / patchcore_service.py)
     python run_patchcore.py --model <model.pt> --json_out <result.json> --input img1.jpg img2.png ...

   Always writes ONE JSON file to --json_out (exit code 0 = success, 1 = failure).
   Success:  {"ok": true, "probability": 0.73, "verdict": "stones_likely" | "no_stones_detected",
              "message": "...", "notes": ["..."], "borderline": false, "threshold": 0.4,
              "n_images": 2, "n_flagged": 1, "findings": [{"label": "cryst", "name": "crystals", "n_images": 1}],
              "images": [{"image": "img1.jpg", "score": 0.8, "threshold": 0.4,
                          "probability_stone": 0.9, "flagged": true, "in_focus": true,
                          "findings": ["cryst", "leuko"], "label_probabilities": {...}}, ...]}
   Failure:  {"ok": false, "error": "FileNotFoundError: ..."}
   The "images" list is in the SAME ORDER as --input.

About the "probability"
-----------------------
The model predicts, per image, the chance of each finding (crystals, leukocytes, ...).
"probability_stone" is the crystal probability re-centred on the model's tuned threshold
(crystal probability == threshold -> 0.5). It is a risk indicator, not a calibrated probability.
This is a screening aid, NOT a diagnosis.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Guarded so that, in web mode, a missing package is reported in the JSON result.
try:
    import numpy as np
    import torch
    from train_patchcore import (STONE_LABEL, build_model, build_transform, focus_score,
                                 list_images, load_pil)
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_ERROR = f"{type(e).__name__}: {e}"


# =====================================================================
#  SETTINGS  -  EDIT THIS SECTION, THEN PRESS "RUN" IN VS CODE
# =====================================================================
MODEL_PATH = r"C:\path\to\save_model_here\patchcore_model.pt"   # created by train_patchcore.py
INPUT_PATH = r"C:\path\to\images_to_classify"                  # one image OR a folder
RESULTS_DIR = r"C:\path\to\results"                            # predictions.csv goes here
THRESHOLD = None            # None = use the crystal threshold saved with the model
SAVE_HEATMAPS = False       # kept for compatibility; this classifier does not make heatmaps
BATCH_SIZE = 8

# Result wording / decision rules for the web app
ALERT_PROBABILITY = 0.5     # overall probability at or above this -> "kidney stones likely"
BORDERLINE_BAND = 0.10      # within +/- this of ALERT_PROBABILITY -> the message says "borderline"
# =====================================================================

LABEL_NAMES = {"cast": "casts", "cryst": "crystals", "epith": "epithelial cells",
               "eryth": "red blood cells", "leuko": "white blood cells", "mycete": "yeast/fungi"}


# ------------------------------------------------------------------ core functions
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        raise ValueError("this model file is from the old PatchCore trainer; retrain with the new train_patchcore.py")
    model = build_model(ck["arch"], len(ck["labels"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    ck["model"] = model.to(device).eval()
    ck["tfm"] = build_transform(tuple(ck["size_hw"]))
    return ck


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


def analyze(paths, ck, device, threshold=None, batch_size=8):
    """Score every image. Returns (list of per-image dicts, None)."""
    labels = list(ck["labels"])
    stone = ck.get("stone_label", STONE_LABEL)
    si = labels.index(stone)
    thrs = dict(ck["thresholds"])
    if threshold is not None:
        thrs[stone] = float(threshold)
    thr_stone = float(thrs[stone])

    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        batch = torch.stack([ck["tfm"](load_pil(p)) for p in chunk]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(ck["model"](batch).float()).cpu().numpy()
        for p, pr in zip(chunk, probs):
            pr = [float(x) for x in pr]
            if not all(math.isfinite(x) for x in pr):
                raise ValueError(f"the model gave no usable score for {Path(p).name}")
            # crystal probability re-centred so that probability == threshold  ->  0.5
            p_stone = _sigmoid(_logit(pr[si]) - _logit(thr_stone))
            sharp = focus_score(p) >= ck.get("focus_min", 0.0)
            out.append({
                "image": Path(p).name, "score": pr[si], "threshold": thr_stone,
                "probability_stone": round(float(p_stone), 4),
                "flagged": bool(p_stone >= ALERT_PROBABILITY),
                "in_focus": bool(sharp),
                "findings": [l for l, x in zip(labels, pr) if x >= thrs[l]],
                "label_probabilities": {l: round(x, 4) for l, x in zip(labels, pr)},
            })
    return out, None


def summarize(items):
    """Combine per-image results into ONE answer for the whole sample.
    A urine sample is judged by its most abnormal field, so the overall
    probability is the maximum over the submitted images."""
    if not items:
        raise ValueError("no images were analysed")
    n = len(items)
    prob = max(i["probability_stone"] for i in items)
    flagged = sum(i["flagged"] for i in items)
    blurry = sum(not i["in_focus"] for i in items)
    likely = prob >= ALERT_PROBABILITY
    pct = round(prob * 100)
    borderline = abs(prob - ALERT_PROBABILITY) <= BORDERLINE_BAND

    counts = {}
    for it in items:
        for lab in it.get("findings", []):
            counts[lab] = counts.get(lab, 0) + 1
    findings = [{"label": l, "name": LABEL_NAMES.get(l, l), "n_images": c} for l, c in counts.items()]
    others = [f for f in findings if f["label"] != STONE_LABEL]

    if likely:
        msg = (f"Kidney stones are likely. Crystals consistent with stone-forming material were found in "
               f"{flagged} of {n} image(s). Probability score: {pct}%.")
    else:
        msg = (f"No signs of kidney stones were found in the {n} submitted image(s). "
               f"Probability score: {pct}%.")
    notes = []
    if borderline:
        notes.append("The score is close to the decision line, so treat the result as borderline and repeat the test.")
    if others:
        notes.append("Other elements also detected: " + ", ".join(
            f"{f['name']} ({f['n_images']}/{n} image(s))" for f in others) + ".")
    if blurry:
        notes.append(f"{blurry} image(s) may be out of focus, which can make the result less reliable.")
    notes.append("This is a screening aid, not a medical diagnosis. Please confirm with a doctor and a laboratory test.")

    return {
        "probability": round(float(prob), 4), "verdict": "stones_likely" if likely else "no_stones_detected",
        "message": msg, "notes": notes, "n_images": n, "n_flagged": int(flagged),
        "borderline": bool(borderline), "findings": findings, "images": items,
    }


def collect_paths(inputs):
    paths = []
    for x in inputs:
        paths += list_images(x)
    return paths


def write_json(path, obj):
    """Strict JSON (no NaN/Infinity) so server.js can always parse it."""
    try:
        text = json.dumps(obj, ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        text = json.dumps({"ok": False, "error": f"Could not encode the result: {e}"})
    Path(path).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ mode B: web app / JSON
def run_json(a):
    """Web app mode. Always writes the JSON result file and returns the exit code."""
    try:
        if IMPORT_ERROR:
            raise RuntimeError(f"a required Python package could not be loaded ({IMPORT_ERROR}). "
                               "Install the packages and keep train_patchcore.py next to run_patchcore.py")
        if not Path(a.model).is_file():
            raise FileNotFoundError(f"model file not found: {a.model}")

        # Use exactly the files server.js sent, in that order (results are matched by position).
        paths = [Path(p) for p in a.input]
        if not paths:
            raise ValueError("no images were received")
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("image file(s) not found: " + ", ".join(missing))
        for p in paths:
            load_pil(p)  # fail now if a picture cannot be decoded

        device = get_device()
        ck = load_model(a.model, device)
        items, _ = analyze(paths, ck, device, a.threshold, a.batch_size)
        result = summarize(items)
        result["threshold"] = items[0]["threshold"]
        result["ok"] = True
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    write_json(a.json_out, result)
    return 0 if result["ok"] else 1


# ------------------------------------------------------------------ mode A: VS Code / batch
def run_batch(a):
    device = get_device()
    ck = load_model(a.model, device)
    paths = collect_paths(a.input)
    if not paths:
        raise SystemExit(f"No images found in {a.input}")
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    items, _ = analyze(paths, ck, device, a.threshold, a.batch_size)
    print(f"{len(paths)} image(s) | crystal threshold = {items[0]['threshold']:.4f} | device = {device}")

    rows = []
    for p, it in zip(paths, items):
        label = "STONE_LIKELY" if it["flagged"] else "NO_STONE"
        rows.append((str(p), f"{it['score']:.4f}", f"{it['threshold']:.4f}", label,
                     f"{it['probability_stone']:.3f}", " ".join(it["findings"]),
                     "ok" if it["in_focus"] else "POSSIBLY_OUT_OF_FOCUS"))
        print(f"{p.name}: crystal_p={it['score']:.3f} -> {label} | found: {', '.join(it['findings']) or 'nothing'}"
              + ("" if it["in_focus"] else "  [possibly out of focus]"))

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "crystal_probability", "crystal_threshold", "label", "probability_stone",
                    "findings", "focus_check"])
        w.writerows(rows)
    summary = summarize(items)
    print(f"\n{sum(r[3] == 'STONE_LIKELY' for r in rows)}/{len(rows)} flagged. Results saved to {out_dir}")
    print(summary["message"])
    for note in summary["notes"]:
        print(" -", note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--input", nargs="+", default=[INPUT_PATH], help="image files and/or folders")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no_heatmaps", action="store_true", default=not SAVE_HEATMAPS)  # accepted, ignored
    ap.add_argument("--json_out", default=None, help="web app mode: write the result to this JSON file")
    a, _ = ap.parse_known_args()

    if a.json_out:
        sys.exit(run_json(a))

    if IMPORT_ERROR:
        raise SystemExit(f"Missing package: {IMPORT_ERROR}\nInstall the packages and keep train_patchcore.py in this folder.")
    if not Path(a.model).exists():
        raise SystemExit(f"MODEL_PATH not found: {a.model}\nEdit the SETTINGS section at the top of this file.")
    for p in a.input:
        if not Path(p).exists():
            raise SystemExit(f"INPUT_PATH not found: {p}\nEdit the SETTINGS section at the top of this file.")
    run_batch(a)


if __name__ == "__main__":
    main()
