"""Harder chart generator (v4). Same five classes as generate_charts.py, but each chart
gets a random visual style and subtler misleaders, so the benchmark is no longer trivial.

Usage:
  python3 generate_hard.py <out_dir> <n_tables> [seed]

Classes (same labels as generate_charts.py):
  label 1: truncated  - bar chart, baseline set just below the smallest bar (subtle)
           inverted   - bar chart, y-axis reversed, with a varied style
           dual       - bar chart plus a line on a second y-axis with its own scale
  label 0: honest_bar - bar chart with zero baseline, more categories and larger values
           honest_line- line chart with non-zero baseline (hard negative, as before)

Each chart also records its 'variant' (font, size, colours, grid, tick style) in the manifest.
A degraded copy (half size, JPEG q35) is written for every chart.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = sys.argv[1]
N_TABLES = int(sys.argv[2])
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 777
KINDS = ["honest_bar", "honest_line", "truncated", "inverted", "dual"]
LABEL = {"honest_bar": 0, "honest_line": 0, "truncated": 1, "inverted": 1, "dual": 1}
MISLEADER = {"honest_bar": None, "honest_line": None, "truncated": "truncated",
             "inverted": "inverted", "dual": "dual"}
CHART_TYPE = {"honest_bar": "bar", "honest_line": "line", "truncated": "bar",
              "inverted": "bar", "dual": "bar"}
BAR_COLOURS = ["#4C78A8", "#54A24B", "#B279A2", "#9D755D", "#72B7B2", "#EECA3B"]
LINE_COLOURS = ["#F58518", "#E45756", "#FF9DA6", "#79706E"]
FONTS = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]
SIZES = [(4, 3), (6, 4), (7, 4.5), (8, 5)]
DPI = 200


def make_table(rng, kind):
    n = int(rng.integers(4, 10)) if kind == "honest_bar" else int(rng.integers(4, 7))
    labels = [f"Cat{chr(65 + i)}" for i in range(n)]
    if kind == "honest_bar":
        values = rng.integers(40, 400, size=n).astype(float)
    else:
        values = rng.integers(20, 100, size=n).astype(float)
    return labels, values


def random_style(rng):
    return {
        "font": FONTS[int(rng.integers(len(FONTS)))],
        "figsize": SIZES[int(rng.integers(len(SIZES)))],
        "tick_size": int(rng.integers(6, 13)),
        "grid": bool(rng.integers(2)),
        "rotate": int(rng.choice([0, 0, 30])),
        "bar_colour": BAR_COLOURS[int(rng.integers(len(BAR_COLOURS)))],
        "line_colour": LINE_COLOURS[int(rng.integers(len(LINE_COLOURS)))],
        "title": bool(rng.integers(2)),
    }


def render(labels, values, kind, path, rng):
    st = random_style(rng)
    meta = {"ymin": 0.0, "inverted": False, "dual": False, "chart_type": CHART_TYPE[kind]}
    plt.rcParams["font.family"] = st["font"]
    fig, ax = plt.subplots(figsize=st["figsize"])
    x = np.arange(len(labels))
    top = values.max() * 1.12
    if kind == "honest_bar":
        ax.bar(x, values, color=st["bar_colour"], width=0.6)
        ax.set_ylim(0, top)
    elif kind == "honest_line":
        ax.plot(x, values, color=st["line_colour"], marker="o", linewidth=2.5)
        base = float(int(values.min() * 0.7 / 5) * 5)
        ax.set_ylim(base, top)
        meta["ymin"] = base
    elif kind == "truncated":
        # subtle: baseline just under the smallest bar, bars still look like bars
        ax.bar(x, values, color=st["bar_colour"], width=0.6)
        base = float(int(values.min() * float(rng.uniform(0.75, 0.9))))
        ax.set_ylim(base, top)
        meta["ymin"] = base
    elif kind == "inverted":
        ax.bar(x, values, color=st["bar_colour"], width=0.6)
        ax.set_ylim(0, top)
        ax.invert_yaxis()
        meta["inverted"] = True
    elif kind == "dual":
        ax.bar(x, values, color=st["bar_colour"], width=0.6)
        ax.set_ylim(0, top)
        ax2 = ax.twinx()
        trend = values[::-1] * float(rng.uniform(0.3, 0.7)) + float(rng.integers(0, 15))
        ax2.plot(x, trend, color=st["line_colour"], marker="o")
        ax2.set_ylim(0, trend.max() * 1.15)
        meta["dual"] = True
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=st["rotate"], ha="right" if st["rotate"] else "center")
    ax.tick_params(labelsize=st["tick_size"])
    if st["grid"]:
        ax.grid(axis="y", alpha=0.3)
    if st["title"]:
        ax.set_title("Quarterly figures", fontsize=st["tick_size"] + 2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    meta["variant"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in st.items()}
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
        split = "test"
        for kind in KINDS:
            labels, values = make_table(rng, kind)
            fname = f"pair{t:04d}_{kind}.png"
            path = os.path.join(OUT, "images", fname)
            meta = render(labels, values, kind, path, rng)
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
