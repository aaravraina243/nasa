r"""
Eye (swollen optic nerve) local service for the Crew Health web app.

Run it in its OWN terminal, in the same folder as run_eye.py and train_eye.py:
    python eye_service.py --model models\eye_model.pt

It loads the model once, then answers requests from server.js:
    GET  /health   -> {"ok": true, "model_loaded": true}
    POST /analyze  -> body {"files": ["C:\\...\\img1.jpg", ...]}  ->  same JSON that run_eye.py writes
Only listens on 127.0.0.1 and only accepts files inside the temp folder (eye-* directories).
Needs only the packages run_eye.py already needs (no Flask).
"""
import argparse
import json
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import run_eye as rp   # reuses analyze(), summarize(), load_model()

STATE = {"ck": None, "device": None}
TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


def allowed(p: Path) -> bool:
    """Only files that server.js saved in its eye-* temp folders."""
    try:
        r = p.resolve()
    except Exception:
        return False
    return TEMP_ROOT in r.parents and r.parent.name.startswith("eye-") and r.is_file()


def run(files):
    paths = [Path(f) for f in files]
    if not paths:
        raise ValueError("no images were received")
    bad = [p.name for p in paths if not allowed(p)]
    if bad:
        raise ValueError("file(s) not allowed or not found: " + ", ".join(bad))
    for p in paths:
        rp.load_pil(p)
    items, _ = rp.analyze(paths, STATE["ck"], STATE["device"], None, rp.BATCH_SIZE)
    result = rp.summarize(items)
    result["threshold"] = items[0]["threshold"]
    result["ok"] = True
    return result


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "model_loaded": STATE["ck"] is not None})
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/analyze":
            return self._send(404, {"ok": False, "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            files = json.loads(self.rfile.read(n) or b"{}").get("files", [])
            result = run(files)
            print(f"analysed {len(files)} image(s): probability={result['probability']}", flush=True)
            self._send(200, result)
        except Exception as e:
            traceback.print_exc()          # the full error shows in THIS terminal
            self._send(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=rp.MODEL_PATH)
    ap.add_argument("--port", type=int, default=5002)
    a = ap.parse_args()

    if rp.IMPORT_ERROR:
        sys.exit(f"Could not load run_eye.py or train_eye.py: {rp.IMPORT_ERROR}")
    if not Path(a.model).is_file():
        sys.exit(f"Model file not found: {a.model}")

    STATE["device"] = rp.get_device()
    print(f"Loading model on {STATE['device']} ...", flush=True)
    STATE["ck"] = rp.load_model(a.model, STATE["device"])
    print(f"Ready. Listening on http://127.0.0.1:{a.port}  (leave this window open)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
