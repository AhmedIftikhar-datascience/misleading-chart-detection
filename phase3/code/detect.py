"""Stage A (extraction) and Stage B (design rules) for bar/line charts.

Pipeline implemented here = the non-VLM part of the Phase 2 method:
  Stage A: chart type from palette pixels; Tesseract reads numeric tick labels
           in the left and right margins, keeping each token's vertical position.
  Stage B: deterministic checks on the extracted evidence:
           truncated baseline (bar charts only), inverted axis, dual axis.
           Abstain when too few left-axis ticks are readable.
The VLM review stage is NOT implemented in this run (see report, section 5).
"""
import re

import numpy as np
import pytesseract
from PIL import Image

CROP = 0.22
MIN_TICKS = 3
BAR_RGB = np.array([76, 120, 168])      # bar colour used by the generator
BAR_TOL = 30
MIN_BAR_PX = 2000                       # bar pixels needed to call a chart "bar"
NUM = re.compile(r"^-?\d+(\.\d+)?$")
RULES = ("truncated", "inverted", "dual")


def _ticks(img, box):
    crop = img.crop(box)
    d = pytesseract.image_to_data(
        crop,
        config="--psm 6 -c tessedit_char_whitelist=0123456789.-",
        output_type=pytesseract.Output.DICT,
    )
    out = []
    for i, txt in enumerate(d["text"]):
        t = txt.strip()
        if t and NUM.match(t):
            out.append((float(t), d["top"][i] + d["height"][i] / 2))
    out.sort(key=lambda v: v[1])
    dedup = []
    for v, y in out:
        if not dedup or abs(y - dedup[-1][1]) > 5:
            dedup.append((v, y))
    return dedup


def chart_type(arr):
    """Bar if enough pixels match the bar colour, otherwise line."""
    diff = np.abs(arr.astype(int) - BAR_RGB).max(axis=2)
    return "bar" if int((diff < BAR_TOL).sum()) >= MIN_BAR_PX else "line"


def bar_bottom(arr):
    """Lowest pixel row holding bar colour (bars are drawn from the baseline)."""
    diff = np.abs(arr.astype(int) - BAR_RGB).max(axis=2)
    rows = np.where((diff < BAR_TOL).sum(axis=1) > 0)[0]
    return float(rows.max()) if len(rows) else None


def extract(path):
    """Stage A: OCR-based axis evidence for one chart image."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img)
    left = _ticks(img, (0, 0, int(w * CROP), h))
    right = _ticks(img, (int(w * (1 - CROP)), 0, w, h))
    return {"left": left, "right": right, "chart_type": chart_type(arr),
            "bar_bottom": bar_bottom(arr)}


def _slope_sign(ticks):
    """Correlation sign of tick value vs pixel y. Normal axis: negative
    (large values sit high). Inverted axis: positive."""
    vals = [v for v, _ in ticks]
    ys = [y for _, y in ticks]
    if len(set(vals)) < 2:
        return 0
    mv, my = sum(vals) / len(vals), sum(ys) / len(ys)
    cov = sum((v - mv) * (y - my) for v, y in ticks)
    return (cov > 0) - (cov < 0)


MISSING_TICK_GAP = 0.5   # bars reaching this fraction of a tick step below the
                         # lowest read tick imply an unread tick (abstain)


def missing_tick(ev):
    """True if bars extend about a full tick step below the lowest read tick,
    so the tick at the baseline was probably missed by OCR."""
    left = sorted(ev["left"], key=lambda t: t[1])
    if len(left) < 2 or ev.get("bar_bottom") is None:
        return False
    steps = [left[i + 1][1] - left[i][1] for i in range(len(left) - 1)]
    step_px = float(np.median(steps))
    if step_px <= 0:
        return False
    gap = (ev["bar_bottom"] - left[-1][1]) / step_px
    return gap >= MISSING_TICK_GAP


def apply_rules(ev, enabled=RULES, gate_truncation=True, missing_check=False):
    """Stage B on OCR evidence. Returns (flagged, fired, abstained)."""
    left = ev["left"]
    if len(left) < MIN_TICKS:
        return False, [], True
    fired = []
    is_bar = ev.get("chart_type") == "bar"
    abstained = False
    if "truncated" in enabled and min(v for v, _ in left) > 0:
        if missing_check and missing_tick(ev):
            abstained = True
        elif (is_bar or not gate_truncation):
            fired.append("truncated")
    if "inverted" in enabled and _slope_sign(left) > 0:
        fired.append("inverted")
    if "dual" in enabled and len(ev["right"]) >= MIN_TICKS:
        fired.append("dual")
    # abstain means "no decision": only when the chart is not flagged
    return bool(fired), fired, (abstained and not fired)


def oracle_rules(axis, ctype, enabled=RULES, gate_truncation=True):
    """Upper bound: the same rules on the generator's true axis settings."""
    fired = []
    if "truncated" in enabled and axis["ymin"] > 0:
        if ctype == "bar" or not gate_truncation:
            fired.append("truncated")
    if "inverted" in enabled and axis["inverted"]:
        fired.append("inverted")
    if "dual" in enabled and axis["dual"]:
        fired.append("dual")
    return bool(fired), fired, False
