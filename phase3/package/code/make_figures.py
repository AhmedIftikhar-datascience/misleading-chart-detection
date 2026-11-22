"""Figures for the Phase 3 report. usage: python3 make_figures.py <results_name> <fig_prefix>"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

name, prefix = sys.argv[1], sys.argv[2]
R = json.load(open(f"/home/claude/phase3/results/{name}.json"))["summary"]
FIG = "/home/claude/phase3/figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

V3C = "Proposed v3: OCR + rules + missing-tick check (clean)"
V3D = "Proposed v3: OCR + rules + missing-tick check (degraded)"
V2D = "Proposed v2: OCR + rules, no missing-tick check (degraded)"
ORA = "Oracle metadata + rules (upper bound)"
ALW = "Baseline B0: always flag"

# Fig 1: precision vs prevalence with CI
prev = [0.05, 0.10, 0.20, 0.50, 0.80, 0.95]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for arm, lab, col, ls in [(V3C, "Proposed v3, clean", "#4C78A8", "-"),
                          (V3D, "Proposed v3, degraded", "#F58518", "-"),
                          (V2D, "Proposed v2, degraded", "#E45756", "--"),
                          (ORA, "Oracle metadata (upper bound)", "#54A24B", ":"),
                          (ALW, "Always flag (baseline)", "#888888", "-.")]:
    pb = R[arm]["precision_by_prevalence"]
    m = [pb[str(p)]["mean"] for p in prev]
    lo = [pb[str(p)]["ci"][0] for p in prev]
    hi = [pb[str(p)]["ci"][1] for p in prev]
    ax.plot([p * 100 for p in prev], m, ls, color=col, marker="o", label=lab)
    ax.fill_between([p * 100 for p in prev], lo, hi, color=col, alpha=0.12)
ax.set_xlabel("Misleading charts in the input stream (%)")
ax.set_ylabel("Precision (share of flags that are correct)")
ax.set_ylim(0, 1.02)
ax.set_title("Precision falls at low prevalence; FPR does not change")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/{prefix}_precision_prevalence.png", dpi=200); plt.close(fig)

# Fig 2: FPR on honest bar vs line charts, and TPR per misleader type
arms = [V3C, V3D, V2D]
labs = ["v3 clean", "v3 degraded", "v2 degraded"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
x = np.arange(len(arms)); w = 0.38
axes[0].bar(x - w / 2, [R[a]["FPR_honest_bar"] for a in arms], w, label="honest bar charts", color="#4C78A8")
axes[0].bar(x + w / 2, [R[a]["FPR_honest_line"] for a in arms], w, label="honest line charts", color="#F58518")
axes[0].set_xticks(x); axes[0].set_xticklabels(labs); axes[0].set_ylabel("False-positive rate")
axes[0].set_ylim(0, 1); axes[0].legend(frameon=False, fontsize=8); axes[0].set_title("False alarms on honest charts")
types = ["truncated", "inverted", "dual"]
for i, a in enumerate([V3C, V3D]):
    axes[1].bar(np.arange(3) + (i - 0.5) * w, [R[a]["recall_by_type"][t] for t in types], w,
                label=["v3 clean", "v3 degraded"][i], color=["#4C78A8", "#F58518"][i])
axes[1].set_xticks(np.arange(3)); axes[1].set_xticklabels(types); axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Recall (share of misleading charts flagged)"); axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("Recall by misleader type")
fig.tight_layout(); fig.savefig(f"{FIG}/{prefix}_fpr_and_recall.png", dpi=200); plt.close(fig)

# Fig 3: ablations (F1 on test)
abl = [("Full v3 (clean)", V3C),
       ("No truncation rule", "Ablation v3: no truncation rule (clean)"),
       ("No inversion rule", "Ablation v3: no inversion rule (clean)"),
       ("No dual-axis rule", "Ablation v3: no dual-axis rule (clean)"),
       ("Truncation ungated", "Ablation v3: truncation rule ungated (clean)"),
       ("Oracle metadata", ORA)]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
vals_f1 = [R[a][1] if False else R[a]["F1"] for _, a in abl]
vals_fpr = [R[a]["FPR"] for _, a in abl]
x = np.arange(len(abl))
ax.bar(x - w / 2, vals_f1, w, label="F1", color="#4C78A8")
ax.bar(x + w / 2, vals_fpr, w, label="False-positive rate", color="#E45756")
ax.set_xticks(x); ax.set_xticklabels([l for l, _ in abl], rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8)
ax.set_title("Ablation: each rule's contribution (clean charts)")
fig.tight_layout(); fig.savefig(f"{FIG}/{prefix}_ablation.png", dpi=200); plt.close(fig)
print("figures written with prefix", prefix)
