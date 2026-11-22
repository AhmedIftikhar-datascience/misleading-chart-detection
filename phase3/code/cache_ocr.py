"""Compute OCR evidence (clean + degraded) for one dataset root and cache it.
usage: python3 cache_ocr.py <data_root> <cache_prefix>"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from detect import bar_bottom, extract  # noqa: E402

root, prefix = sys.argv[1], sys.argv[2]
RES = "/home/claude/phase3/results"
m = json.load(open(f"{root}/manifest.json"))
for key, name in (("file", "clean"), ("file_degraded", "degraded")):
    out = {}
    for r in m:
        ev = extract(f"{root}/{r[key]}")
        ev["bar_bottom"] = bar_bottom(np.asarray(Image.open(f"{root}/{r[key]}").convert("RGB")))
        out[r[key]] = ev
    json.dump(out, open(f"{RES}/{prefix}_{name}.json", "w"))
    print("done", name, len(out), flush=True)
