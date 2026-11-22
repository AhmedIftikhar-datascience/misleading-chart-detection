"""Run a free Gemini model as the image-only VLM baseline (B1) on the test split.

Usage (run on YOUR computer, with your own keys):
  Windows (Command Prompt):  set GEMINI_API_KEY=key_one
                             set GEMINI_API_KEY_2=key_two
  Mac / Linux:               export GEMINI_API_KEY=key_one
                             export GEMINI_API_KEY_2=key_two
  python3 run_vlm.py --manifest sample_manifest.json --out vlm_clean.jsonl
  python3 run_vlm.py --manifest sample_manifest.json --out vlm_degraded.jsonl --condition degraded

Options:
  --model        default gemini-3.5-flash-lite (change to a model name shown in AI Studio if 404)
  --per-kind     charts per class (default 6 -> 30 charts per condition)
  --condition    clean (default) or degraded
  --timeout      seconds to wait per request (default 240)
  --dry-run      build the sample and print the first prompt, without calling the API

Keys: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 and GEMINI_API_KEY_4 are read if set. When one key hits
its free-tier quota, the script switches to the next key. If all keys are used up, it stops;
run the same command again later and it resumes from the last successful chart.

The script writes one JSON line per chart and resumes if re-run with the same --out file.
Keys are read from the environment only; they are never written to disk or printed.
"""
import argparse
import base64
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request

PROMPT = (
    "You are checking a chart for design choices that mislead the reader. "
    "Look only at the chart. Possible misleaders: "
    "'truncated' = a bar chart whose value axis does not start at zero; "
    "'inverted' = the value axis runs in reverse, so larger values sit lower; "
    "'dual' = a second value axis on the right with its own scale; "
    "'none' = none of these (for example, a line chart whose axis starts above zero is fine). "
    "Answer with JSON only: {\"misleading\": true or false, \"misleader\": one of "
    "\"truncated\", \"inverted\", \"dual\", \"none\"}."
)
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KEY_NAMES = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]


class QuotaError(Exception):
    """This key has used its free-tier quota (HTTP 429 / RESOURCE_EXHAUSTED)."""


class ModelMissing(Exception):
    """The model name is not available to this key (HTTP 404)."""


def sample(manifest, per_kind, seed):
    rng = random.Random(seed)
    test = [r for r in manifest if r["split"] == "test"]
    out = []
    for kind in ("honest_bar", "honest_line", "truncated", "inverted", "dual"):
        pool = [r for r in test if r["kind"] == kind]
        out.extend(rng.sample(pool, min(per_kind, len(pool))))
    return out


def ask(model, key, image_path, timeout, retries=4):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/jpeg" if image_path.lower().endswith(".jpg") else "image/png"
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": mime, "data": b64}},
        ]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        URL.format(model=model),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code == 404:
                raise ModelMissing(f"HTTP 404: {detail}")
            if e.code == 429 or "RESOURCE_EXHAUSTED" in detail:
                # per-minute limits clear after a short wait; daily quota does not
                if attempt < retries - 1 and "per minute" in detail.lower():
                    time.sleep(30)
                    continue
                raise QuotaError(f"HTTP {e.code}: {detail}")
            last = RuntimeError(f"HTTP {e.code}: {detail}")
            if e.code in (500, 503) and attempt < retries - 1:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise last
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                KeyError, IndexError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"network/timeout after {retries} tries: {type(e).__name__}")
    raise last


def parse(text):
    obj = json.loads(text)
    mis = bool(obj.get("misleading"))
    m = obj.get("misleader", "none")
    if m not in ("truncated", "inverted", "dual", "none"):
        m = "none"
    return int(mis), m


def load_keys():
    keys = []
    for name in KEY_NAMES:
        val = os.environ.get(name, "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default=None,
                    help="folder that contains the manifest's image paths (default: manifest folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--per-kind", type=int, default=6)
    ap.add_argument("--condition", choices=["clean", "degraded"], default="clean")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = a.image_root or os.path.dirname(os.path.abspath(a.manifest))
    manifest = json.load(open(a.manifest))
    rows = sample(manifest, a.per_kind, a.seed)
    key_name = "file" if a.condition == "clean" else "file_degraded"
    print(f"sample: {len(rows)} charts, condition={a.condition}, model={a.model}")

    if a.dry_run:
        print("PROMPT:\n" + PROMPT)
        print("first chart:", rows[0]["pair_id"], rows[0]["kind"], os.path.join(root, rows[0][key_name]))
        return

    keys = load_keys()
    if not keys:
        sys.exit("GEMINI_API_KEY is not set. See the instructions in the reply.")
    print(f"keys available: {len(keys)}")
    key_idx = 0

    done = set()  # only SUCCESSFUL answers are skipped on resume; errors are retried
    if os.path.exists(a.out):
        for line in open(a.out):
            rec_old = json.loads(line)
            if rec_old.get("error") is None:
                done.add(rec_old["file"])

    with open(a.out, "a") as out:
        for i, r in enumerate(rows, 1):
            if r[key_name] in done:
                continue
            rec = {"pair_id": r["pair_id"], "kind": r["kind"], "label": r["label"],
                   "misleader": r["misleader"], "file": r[key_name], "model": a.model,
                   "condition": a.condition}
            while True:
                if key_idx >= len(keys):
                    out.flush()
                    print(f"All {len(keys)} key(s) have used their free quota for now. "
                          "Run the same command again later; it resumes from here.")
                    print("finished (stopped early):", a.out)
                    return
                try:
                    text = ask(a.model, keys[key_idx], os.path.join(root, r[key_name]), a.timeout)
                    pred, mis = parse(text)
                    rec.update({"pred": pred, "pred_misleader": mis, "raw": text, "error": None})
                    break
                except QuotaError:
                    key_idx += 1
                    print(f"key {key_idx} used up; switching to key {key_idx + 1}" if key_idx < len(keys)
                          else "no keys left", flush=True)
                    continue
                except ModelMissing as e:
                    out.write(json.dumps({**rec, "pred": None, "pred_misleader": None,
                                          "raw": None, "error": str(e)[:300]}) + "\n")
                    sys.exit("Model not available (HTTP 404). Check the model list in AI Studio and rerun with "
                             "--model NAME. Failed lines are retried on resume.")
                except Exception as e:  # keep going; errors are counted in scoring
                    rec.update({"pred": None, "pred_misleader": None, "raw": None,
                                "error": str(e)[:300]})
                    break
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{i}/{len(rows)} {r['kind']} pred={rec['pred']} err={rec['error'] is not None}", flush=True)
            time.sleep(1.5)  # be gentle with the free tier
    print("finished:", a.out)


if __name__ == "__main__":
    main()
