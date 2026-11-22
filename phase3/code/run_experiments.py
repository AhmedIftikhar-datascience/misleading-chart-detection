"""Phase 3 experiments (v2). Writes /home/claude/phase3/results/*.json|csv.

Conditions: clean charts and degraded charts (downscaled + JPEG q35).
Arms: proposed OCR+rules (gated and ungated truncation), oracle-metadata upper bound,
rule ablations, and trivial baselines. Test split only for reported numbers;
dev split is reported for reference and was used only to check the pipeline runs.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from detect import RULES, apply_rules, extract, oracle_rules  # noqa: E402

BASE = "/home/claude/phase3"
DATA = f"{BASE}/data"
RES = f"{BASE}/results"
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(7)
PREV = [0.05, 0.10, 0.20, 0.50, 0.80, 0.95]
N_SAMPLE, N_BOOT = 400, 1000

manifest = json.load(open(f"{DATA}/manifest.json"))


def cached_ocr(name, key):
    path = f"{RES}/ocr_{name}.json"
    if os.path.exists(path):
        return json.load(open(path))
    out = {r[key]: extract(f"{DATA}/{r[key]}") for r in manifest}
    json.dump(out, open(path, "w"))
    return out


ocr_clean = cached_ocr("clean", "file")
ocr_deg = cached_ocr("degraded", "file_degraded")


def mk(ocr, key, **kw):
    return lambda r: apply_rules(ocr[r[key]], **kw)


def orc(**kw):
    return lambda r: oracle_rules(r["axis"], r["chart_type"], **kw)


ARMS = {
    "Proposed: OCR + rules, clean": mk(ocr_clean, "file"),
    "Proposed: OCR + rules, degraded": mk(ocr_deg, "file_degraded"),
    "Oracle metadata + rules (upper bound)": orc(),
    "Ablation: OCR, truncation rule ungated": mk(ocr_clean, "file", gate_truncation=False),
    "Ablation: oracle, truncation rule ungated": orc(gate_truncation=False),
    "Ablation: OCR, no truncation rule": mk(ocr_clean, "file", enabled=("inverted", "dual")),
    "Ablation: OCR, no inversion rule": mk(ocr_clean, "file", enabled=("truncated", "dual")),
    "Ablation: OCR, no dual-axis rule": mk(ocr_clean, "file", enabled=("truncated", "inverted")),
    "Baseline B0: always flag": lambda r: (True, [], False),
    "Baseline B0b: never flag": lambda r: (False, [], False),
}
P = "Proposed: OCR + rules, clean"
ORACLE = "Oracle metadata + rules (upper bound)"


def conf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def rates(tp, fp, fn, tn):
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if tp else 0.0
    return tpr, fpr, prec, f1


def ci(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def evaluate(rows, arm_fn):
    y = np.array([r["label"] for r in rows])
    dec = [arm_fn(r) for r in rows]
    p = np.array([int(d[0]) for d in dec])
    abst = np.array([d[2] for d in dec])
    return y, p, abst, dec


test = [r for r in manifest if r["split"] == "test"]
test_pairs = sorted({r["pair_id"] for r in test})
pair_idx = {q: [i for i, r in enumerate(test) if r["pair_id"] == q] for q in test_pairs}
kinds = np.array([r["kind"] for r in test])
y_all = np.array([r["label"] for r in test])

summary, preds = {}, {}
for arm, fn in ARMS.items():
    y, p, abst, _ = evaluate(test, fn)
    preds[arm] = p
    tp, fp, fn_, tn = conf(y, p)
    tpr, fpr, prec, f1 = rates(tp, fp, fn_, tn)
    b_tpr, b_fpr, b_f1 = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(test_pairs, size=len(test_pairs), replace=True)
        idx = np.array([i for q in pick for i in pair_idx[q]])
        bt, bf, _, bf1 = rates(*conf(y[idx], p[idx]))
        b_tpr.append(bt); b_fpr.append(bf); b_f1.append(bf1)
    # false-positive rate split by honest chart type: the key stratification
    fpr_bar = float(p[kinds == "honest_bar"].mean())
    fpr_line = float(p[kinds == "honest_line"].mean())
    recall_type = {k: float(p[kinds == k].mean()) for k in ("truncated", "inverted", "dual")}
    # precision at each prevalence: resample 400 charts with replacement
    pos_f, neg_f = p[y == 1], p[y == 0]
    prec_by_p = {}
    for pv in PREV:
        npos = int(round(pv * N_SAMPLE))
        vals = []
        for _ in range(N_BOOT):
            a = pos_f[rng.integers(0, len(pos_f), npos)].sum()
            b = neg_f[rng.integers(0, len(neg_f), N_SAMPLE - npos)].sum()
            vals.append(a / (a + b) if a + b else 0.0)
        prec_by_p[str(pv)] = {"mean": float(np.mean(vals)), "ci": ci(vals)}
    summary[arm] = {
        "TP": tp, "FP": fp, "FN": fn_, "TN": tn,
        "TPR": tpr, "TPR_CI": ci(b_tpr), "FPR": fpr, "FPR_CI": ci(b_fpr),
        "FPR_honest_bar": fpr_bar, "FPR_honest_line": fpr_line,
        "F1": f1, "F1_CI": ci(b_f1), "precision_observed": prec,
        "abstain_misleading": float(abst[y == 1].mean()),
        "abstain_honest": float(abst[y == 0].mean()),
        "recall_by_type": recall_type,
        "precision_by_prevalence": prec_by_p,
    }

def mcnemar(a, b):
    ca = preds[a] == y_all; cb = preds[b] == y_all
    b01 = int((ca & ~cb).sum()); b10 = int((~ca & cb).sum())
    if b01 + b10 == 0:
        return {"b": b01, "c": b10, "p": 1.0}
    chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return {"b": b01, "c": b10, "p": float(1 - stats.chi2.cdf(chi, 1))}

tests = {
    "clean vs degraded": mcnemar(P, "Proposed: OCR + rules, degraded"),
    "clean vs oracle": mcnemar(P, ORACLE),
    "clean vs ungated truncation": mcnemar(P, "Ablation: OCR, truncation rule ungated"),
    "clean vs always-flag": mcnemar(P, "Baseline B0: always flag"),
}

# extraction diagnostics on all splits
diag = {}
for name, ocr, key in (("clean", ocr_clean, "file"), ("degraded", ocr_deg, "file_degraded")):
    rows = {}
    for r in manifest:
        e = ocr[r[key]]
        rows.setdefault(r["kind"], []).append((len(e["left"]) >= 3, e["chart_type"] == r["chart_type"]))
    diag[name] = {k: {"readable_rate": float(np.mean([a for a, _ in v])),
                      "chart_type_acc": float(np.mean([b for _, b in v]))} for k, v in rows.items()}

json.dump({"n_test_charts": len(test), "n_test_pairs": len(test_pairs),
           "summary": summary, "mcnemar": tests, "extraction_diagnostics": diag},
          open(f"{RES}/results_v2.json", "w"), indent=1, default=float)
with open(f"{RES}/arm_table_v2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "TPR", "FPR", "FPR_honest_bar", "FPR_honest_line", "F1",
                "precision@5%", "precision@50%", "precision@95%", "abstain_misleading", "abstain_honest"])
    for arm, s in summary.items():
        pb = s["precision_by_prevalence"]
        w.writerow([arm, f"{s['TPR']:.3f}", f"{s['FPR']:.3f}", f"{s['FPR_honest_bar']:.3f}",
                    f"{s['FPR_honest_line']:.3f}", f"{s['F1']:.3f}",
                    f"{pb['0.05']['mean']:.3f}", f"{pb['0.5']['mean']:.3f}", f"{pb['0.95']['mean']:.3f}",
                    f"{s['abstain_misleading']:.3f}", f"{s['abstain_honest']:.3f}"])
print("test charts", len(test), "pairs", len(test_pairs))
for arm, s in summary.items():
    print(f"{arm:44s} TPR={s['TPR']:.3f} FPR={s['FPR']:.3f} (bar {s['FPR_honest_bar']:.3f}, line {s['FPR_honest_line']:.3f}) "
          f"F1={s['F1']:.3f} prec@5%={s['precision_by_prevalence']['0.05']['mean']:.3f} "
          f"prec@95%={s['precision_by_prevalence']['0.95']['mean']:.3f}")
print(json.dumps(tests, indent=1))
print(json.dumps(diag, indent=1))
