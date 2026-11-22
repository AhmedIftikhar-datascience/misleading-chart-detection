"""Generate paired charts from shared data tables (v2, after the pilot run).

Classes:
  label 1 (misleading): bar chart with truncated baseline | inverted y-axis | dual y-axis
  label 0 (honest):     bar chart with zero baseline (honest_bar)
                        line chart with non-zero baseline (honest_line)  <- hard negative
A line chart with a non-zero baseline is a legitimate design, so a rule that
flags any non-zero baseline should produce false alarms on it.
Each table yields one chart per class kind, so a pair shares its numbers.
A degraded copy (downscaled + JPEG q=35) is written for every chart.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/phase3/data"
N_TABLES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SEED = int(os.environ.get("CHART_SEED", "20261115"))
ALL_TEST = os.environ.get("CHART_ALL_TEST") == "1"
DPI = 200
KINDS = ["honest_bar", "honest_line", "truncated", "inverted", "dual"]
LABEL = {"honest_bar": 0, "honest_line": 0, "truncated": 1, "inverted": 1, "dual": 1}
MISLEADER = {"honest_bar": None, "honest_line": None, "truncated": "truncated",
             "inverted": "inverted", "dual": "dual"}
CHART_TYPE = {"honest_bar": "bar", "honest_line": "line", "truncated": "bar",
              "inverted": "bar", "dual": "bar"}
BAR = "#4C78A8"
LINE = "#F58518"


def make_table(rng):
    n = int(rng.integers(4, 7))
    labels = [f"Cat{chr(65 + i)}" for i in range(n)]
    values = rng.integers(20, 100, size=n).astype(float)
    return labels, values


def render(labels, values, kind, path):
    meta = {"ymin": 0.0, "inverted": False, "dual": False, "chart_type": CHART_TYPE[kind]}
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    top = values.max() * 1.1
    if kind == "honest_bar":
        ax.bar(x, values, color=BAR, width=0.6)
        ax.set_ylim(0, top)
    elif kind == "honest_line":
        ax.plot(x, values, color=LINE, marker="o", linewidth=2.5)
        base = float(int(values.min() * 0.7 / 5) * 5)
        ax.set_ylim(base, top)
        meta["ymin"] = base
    elif kind == "truncated":
        ax.bar(x, values, color=BAR, width=0.6)
        base = float(int(values.max() * 0.6 / 5) * 5)
        ax.set_ylim(base, top)
        meta["ymin"] = base
    elif kind == "inverted":
        ax.bar(x, values, color=BAR, width=0.6)
        ax.set_ylim(0, top)
        ax.invert_yaxis()
        meta["inverted"] = True
    elif kind == "dual":
        ax.bar(x, values, color=BAR, width=0.6)
        ax.set_ylim(0, top)
        ax2 = ax.twinx()
        trend = values[::-1] * 0.5 + 5
        ax2.plot(x, trend, color="#E45756", marker="o")
        ax2.set_ylim(0, trend.max() * 1.1)
        meta["dual"] = True
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return meta


def degrade(src, dst):
    img = Image.open(src).convert("RGB")
    w, h = img.size
    img = img.resize((w // 2, h // 2), Image.BILINEAR)
    img.save(dst, format="JPEG", quality=35)


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "images_degraded"), exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest = []
    for t in range(N_TABLES):
        labels, values = make_table(rng)
        split = "dev" if (t < N_TABLES // 3 and not ALL_TEST) else "test"
        for kind in KINDS:
            fname = f"pair{t:04d}_{kind}.png"
            path = os.path.join(OUT, "images", fname)
            meta = render(labels, values, kind, path)
            dpath = os.path.join(OUT, "images_degraded", fname.replace(".png", ".jpg"))
            degrade(path, dpath)
            manifest.append({
                "pair_id": f"pair{t:04d}", "split": split, "kind": kind,
                "label": LABEL[kind], "misleader": MISLEADER[kind],
                "chart_type": CHART_TYPE[kind],
                "file": f"images/{fname}",
                "file_degraded": f"images_degraded/{fname.replace('.png', '.jpg')}",
                "values": values.tolist(), "categories": labels, "axis": meta,
            })
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest)} charts from {N_TABLES} tables to {OUT}")


if __name__ == "__main__":
    main()
