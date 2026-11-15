r"""
Run the trained eye model (swollen optic nerve / papilledema) on retinal (fundus) photos.

Keep this file in the iter-10 folder, next to train_eye.py and server.js.

The model is a supervised binary classifier (an ensemble of CNNs with flip test-time augmentation)
trained by train_eye.py. Everything the web app sees (command line, JSON, field names) is unchanged.

A) Batch mode (VS Code "Run Python File"): edit INPUT_PATH below. Output: predictions.csv + heatmap PNGs
   (Grad-CAM: the bright areas are the image regions that pushed the score towards "papilledema").
B) Web app mode (started by server.js when the eye service is not running):
     python run_eye.py --model models\eye_model.pt --json_out result.json --input img1.jpg img2.png ...
   It ALWAYS writes one JSON file (even on failure) and exits with 0 (ok) or 1 (failed):
     {"ok": true, "probability": 0.8, "verdict": "papilledema_likely" | "no_papilledema_detected",
      "message": "...", "notes": [...], "borderline": false, "n_images": 2, "n_flagged": 1, "threshold": 1.2,
      "images": [{"image": "a.jpg", "score": 1.4, "threshold": 1.2, "probability_abnormal": 0.9,
                  "flagged": true, "in_focus": true}, ...]}
     {"ok": false, "error": "..."}
   "images" keeps the SAME ORDER as --input (server.js matches them by position).

The probability comes from the calibration done in train_eye.py (0.5 exactly at the threshold).
It is a screening aid, NOT a diagnosis.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    from train_eye import (build_model, build_transform, ensemble_logit, focus_score, list_images,
                           load_pil, probability)
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_ERROR = f"{type(e).__name__}: {e}"

HERE = Path(__file__).resolve().parent

# =====================================================================
#  SETTINGS  (batch mode only; the web app passes everything on the command line)
# =====================================================================
MODEL_PATH = str(HERE / "models" / "eye_model.pt")
INPUT_PATH = r"C:\path\to\eye_images_to_check"      # one image OR a folder
RESULTS_DIR = str(HERE / "eye_results")
THRESHOLD = None            # None = use the threshold saved with the model
SAVE_HEATMAPS = True
BATCH_SIZE = 8
ALERT_PROBABILITY = 0.5     # overall probability at or above this -> "swollen optic nerve likely"
BORDERLINE_BAND = 0.10      # within +/- this of ALERT_PROBABILITY -> "borderline"
# =====================================================================


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dicts" not in ck:
        raise ValueError("this eye_model.pt comes from the old PatchCore trainer; retrain with the new train_eye.py")
    models_ = []
    for sd in ck["state_dicts"]:
        m = build_model(ck["arch"], 1, pretrained=False)
        m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
        models_.append(m.to(device).eval())
    ck["models"] = models_
    ck["tfm"] = build_transform(tuple(ck["size_hw"]))
    return ck


def analyze(paths, ck, device, threshold=None, batch_size=8):
    """Score every image. Returns (per-image dicts, None)."""
    thr = float(threshold if threshold is not None else ck["threshold"])
    w, scale = float(ck["prob_w"]), float(ck["prob_scale"])
    tta = bool(ck.get("tta", False))
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        x = torch.stack([ck["tfm"](load_pil(p)) for p in chunk]).to(device)
        logits = ensemble_logit(ck["models"], x, tta).cpu().numpy()
        for p, s in zip(chunk, logits):
            s = float(s)
            if not math.isfinite(s):
                raise ValueError(f"the model gave no usable score for {Path(p).name}")
            p_abn = probability(s, thr, w, scale)
            out.append({"image": Path(p).name, "score": s, "threshold": thr,
                        "probability_abnormal": round(float(p_abn), 4),
                        "flagged": bool(p_abn >= ALERT_PROBABILITY),
                        "in_focus": bool(focus_score(p) >= ck.get("focus_min", 0.0))})
    return out, None


def summarize(items):
    """One answer for the whole set: judged by the most abnormal image (maximum probability)."""
    if not items:
        raise ValueError("no images were analysed")
    n = len(items)
    prob = max(i["probability_abnormal"] for i in items)
    flagged = sum(i["flagged"] for i in items)
    blurry = sum(not i["in_focus"] for i in items)
    likely = prob >= ALERT_PROBABILITY
    pct = round(prob * 100)
    borderline = abs(prob - ALERT_PROBABILITY) <= BORDERLINE_BAND

    if likely:
        msg = (f"A swollen optic nerve (papilledema) is likely. Signs of optic disc swelling were found in {flagged} of {n} image(s). "
               "Swelling of the optic nerve is often caused by raised pressure inside the skull (intracranial pressure). "
               f"In spaceflight, body fluids shifting towards the head can lead to this. Probability score: {pct}%.")
    else:
        msg = f"No signs of a swollen optic nerve were found in the {n} submitted image(s). Probability score: {pct}%."

    notes = []
    if likely:
        notes.append("A swollen optic nerve can damage vision if it is not treated, so please see an eye doctor or the flight surgeon soon for a full eye examination.")
        notes.append("Some harmless conditions (for example optic disc drusen, called pseudopapilledema) can look similar, so only a specialist can confirm the cause.")
    else:
        notes.append("A normal-looking result does not rule out eye or pressure problems. If you have lasting headaches, blurred or double vision, or brief blackouts of vision, tell a doctor.")
    if borderline:
        notes.append("The score is close to the decision line, so treat the result as borderline and repeat the test with clearer photos.")
    if blurry:
        notes.append(f"{blurry} image(s) may be out of focus, which can make the result less reliable.")
    notes.append("Use sharp, well-lit retinal (fundus) photos centred on the optic disc. Other kinds of pictures give meaningless results.")
    notes.append("This is a screening aid, not a medical diagnosis. Please confirm with an eye doctor.")

    return {"probability": round(float(prob), 4), "verdict": "papilledema_likely" if likely else "no_papilledema_detected",
            "message": msg, "notes": notes, "n_images": n, "n_flagged": int(flagged),
            "borderline": bool(borderline), "images": items}


def collect_paths(inputs):
    paths = []
    for x in inputs:
        paths += list_images(x)
    return paths


# ------------------------------------------------------------------ Grad-CAM heatmap (batch mode only)
def cam_layer(model):
    return model.layer4 if hasattr(model, "layer4") else model.features[-1]


def grad_cam(model, x):
    """Heatmap (h, w) in 0..1 showing what pushed this model's score towards 'papilledema'."""
    store = {}

    def fwd(_m, _i, o):
        store["a"] = o.detach()
        o.register_hook(lambda g: store.__setitem__("g", g.detach()))

    h = cam_layer(model).register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            model(x).squeeze(1).sum().backward()
    finally:
        h.remove()
    cam = torch.relu((store["g"].mean(dim=(2, 3), keepdim=True) * store["a"]).sum(1))[0].cpu().numpy()
    return cam / (cam.max() + 1e-8)


def upsample_map(low_res, size_hw, sigma=4):
    m = F.interpolate(torch.from_numpy(low_res)[None, None], size=tuple(size_hw), mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(m, sigma=sigma)


def write_json(path, obj):
    try:
        text = json.dumps(obj, ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        text = json.dumps({"ok": False, "error": f"Could not encode the result: {e}"})
    Path(path).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ web app mode
def run_json(a):
    try:
        if IMPORT_ERROR:
            raise RuntimeError(f"a required Python package or train_eye.py could not be loaded ({IMPORT_ERROR})")
        if not Path(a.model).is_file():
            raise FileNotFoundError(f"model file not found: {a.model}")
        paths = [Path(p) for p in a.input]          # exactly the files server.js sent, in order
        if not paths:
            raise ValueError("no images were received")
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("image file(s) not found: " + ", ".join(missing))
        for p in paths:
            load_pil(p)                              # fail now if a picture cannot be decoded
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


# ------------------------------------------------------------------ batch mode
def run_batch(a):
    device = get_device()
    ck = load_model(a.model, device)
    paths = collect_paths(a.input)
    if not paths:
        raise SystemExit(f"No images found in {a.input}")
    threshold = a.threshold if a.threshold is not None else ck["threshold"]
    size_hw = tuple(ck["size_hw"])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} image(s) | threshold = {threshold:.4f} | models = {len(ck['models'])} | device = {device}")
    items, _ = analyze(paths, ck, device, a.threshold, a.batch_size)

    if not a.no_heatmaps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    rows = []
    for p, it in zip(paths, items):
        label = "PAPILLEDEMA LIKELY" if it["flagged"] else "NORMAL"
        rows.append((str(p), f"{it['score']:.4f}", f"{threshold:.4f}", label, f"{it['probability_abnormal']:.3f}",
                     "ok" if it["in_focus"] else "POSSIBLY_OUT_OF_FOCUS"))
        print(f"{p.name}: score={it['score']:.4f} p={it['probability_abnormal']:.2f} -> {label}"
              + ("" if it["in_focus"] else "  [possibly out of focus]"))
        if not a.no_heatmaps:
            x = ck["tfm"](load_pil(p))[None].to(device)
            cam = upsample_map(grad_cam(ck["models"][0], x), size_hw)
            img = load_pil(p).resize((size_hw[1], size_hw[0]))
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(img); ax[0].set_title("input"); ax[0].axis("off")
            ax[1].imshow(img)
            ax[1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax[1].set_title(f"score {it['score']:.3f} / threshold {threshold:.3f}"); ax[1].axis("off")
            fig.suptitle(label, fontsize=11)
            fig.savefig(out_dir / f"{p.stem}_result.png", dpi=110, bbox_inches="tight")
            plt.close(fig)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "score", "threshold", "label", "probability_abnormal", "focus_check"])
        w.writerows(rows)
    summary = summarize(items)
    print(f"\n{summary['n_flagged']}/{len(rows)} flagged. Results saved to {out_dir}")
    print(summary["message"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--input", nargs="+", default=[INPUT_PATH], help="image files and/or folders")
    ap.add_argument("--out_dir", default=RESULTS_DIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--no_heatmaps", action="store_true", default=not SAVE_HEATMAPS)
    ap.add_argument("--json_out", default=None, help="web app mode: write the result to this JSON file")
    a, _ = ap.parse_known_args()

    if a.json_out:
        sys.exit(run_json(a))
    if IMPORT_ERROR:
        raise SystemExit(f"Could not load: {IMPORT_ERROR}\nKeep train_eye.py in this folder and install the packages.")
    if not Path(a.model).exists():
        raise SystemExit(f"MODEL_PATH not found: {a.model}\nTrain it first with train_eye.py.")
    for p in a.input:
        if not Path(p).exists():
            raise SystemExit(f"INPUT_PATH not found: {p}\nEdit the SETTINGS section at the top of this file.")
    run_batch(a)


if __name__ == "__main__":
    main()
