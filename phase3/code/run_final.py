"""Final evaluation for one dataset root and its OCR cache.
usage: python3 run_final.py <data_root> <cache_prefix> <out_name>
  cache_prefix: 'ocr' for the original dataset (files ocr_clean.json / ocr_degraded.json)
                'confirm' for the confirmatory set (confirm_clean.json / confirm_degraded.json)
Reports on the test split only."""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from detect import apply_rules, oracle_rules  # noqa: E402

root, prefix, out_name = sys.argv[1], sys.argv[2], sys.argv[3]
RES = "/home/claude/phase3/results"
rng = np.random.default_rng(11)
PREV = [0.05, 0.10, 0.20, 0.50, 0.80, 0.95]
N_SAMPLE, N_BOOT = 400, 1000

manifest = json.load(open(f"{root}/manifest.json"))
test = [r for r in manifest if r["split"] == "test"]
oc = json.load(open(f"{RES}/{prefix}_clean.json"))
od = json.load(open(f"{RES}/{prefix}_degraded.json"))


def V(name, ocr, key, **kw):
    return name, (lambda r: apply_rules(ocr[r[key]], **kw))


ARMS = dict([
    V("Proposed v3: OCR + rules + missing-tick check (clean)", oc, "file", missing_check=True),
    V("Proposed v3: OCR + rules + missing-tick check (degraded)", od, "file_degraded", missing_check=True),
    V("Proposed v2: OCR + rules, no missing-tick check (clean)", oc, "file"),
    V("Proposed v2: OCR + rules, no missing-tick check (degraded)", od, "file_degraded"),
    V("Ablation v3: truncation rule ungated (clean)", oc, "file", missing_check=True, gate_truncation=False),
    V("Ablation v3: no truncation rule (clean)", oc, "file", missing_check=True, enabled=("inverted", "dual")),
    V("Ablation v3: no inversion rule (clean)", oc, "file", missing_check=True, enabled=("truncated", "dual")),
    V("Ablation v3: no dual-axis rule (clean)", oc, "file", missing_check=True, enabled=("truncated", "inverted")),
])
ARMS["Oracle metadata + rules (upper bound)"] = lambda r: oracle_rules(r["axis"], r["chart_type"])
ARMS["Oracle, truncation rule ungated"] = lambda r: oracle_rules(r["axis"], r["chart_type"], gate_truncation=False)
ARMS["Baseline B0: always flag"] = lambda r: (True, [], False)
ARMS["Baseline B0b: never flag"] = lambda r: (False, [], False)

kinds = np.array([r["kind"] for r in test])
y = np.array([r["label"] for r in test])
pairs = sorted({r["pair_id"] for r in test})
pidx = {q: np.array([i for i, r in enumerate(test) if r["pair_id"] == q]) for q in pairs}


def rates(yy, pp):
    tp = int(((pp == 1) & (yy == 1)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
    fn = int(((pp == 0) & (yy == 1)).sum()); tn = int(((pp == 0) & (yy == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if tp else 0.0
    return tpr, fpr, prec, f1


def ci(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


preds, summary = {}, {}
for name, fn in ARMS.items():
    dec = [fn(r) for r in test]
    p = np.array([int(d[0]) for d in dec]); ab = np.array([bool(d[2]) for d in dec])
    preds[name] = p
    tpr, fpr, prec, f1 = rates(y, p)
    bt, bf, bf1 = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(pairs, size=len(pairs), replace=True)
        idx = np.concatenate([pidx[q] for q in pick])
        a, b, _, c = rates(y[idx], p[idx]); bt.append(a); bf.append(b); bf1.append(c)
    pos_f, neg_f = p[y == 1], p[y == 0]
    prec_pv = {}
    for pv in PREV:
        npos = int(round(pv * N_SAMPLE)); vals = []
        for _ in range(N_BOOT):
            a = pos_f[rng.integers(0, len(pos_f), npos)].sum()
            b = neg_f[rng.integers(0, len(neg_f), N_SAMPLE - npos)].sum()
            vals.append(a / (a + b) if a + b else 0.0)
        prec_pv[str(pv)] = {"mean": float(np.mean(vals)), "ci": ci(vals)}
    summary[name] = {
        "TPR": tpr, "TPR_CI": ci(bt), "FPR": fpr, "FPR_CI": ci(bf), "F1": f1, "F1_CI": ci(bf1),
        "FPR_honest_bar": float(p[kinds == "honest_bar"].mean()),
        "FPR_honest_line": float(p[kinds == "honest_line"].mean()),
        "recall_by_type": {k: float(p[kinds == k].mean()) for k in ("truncated", "inverted", "dual")},
        "abstain_misleading": float(ab[y == 1].mean()), "abstain_honest": float(ab[y == 0].mean()),
        "precision_by_prevalence": prec_pv,
    }


def mcnemar(a, b):
    ca = preds[a] == y; cb = preds[b] == y
    b01 = int((ca & ~cb).sum()); b10 = int((~ca & cb).sum())
    if b01 + b10 == 0:
        return {"b": b01, "c": b10, "p": 1.0}
    chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return {"b": b01, "c": b10, "p": float(1 - stats.chi2.cdf(chi, 1))}


V3C, V3D = "Proposed v3: OCR + rules + missing-tick check (clean)", "Proposed v3: OCR + rules + missing-tick check (degraded)"
V2C, V2D = "Proposed v2: OCR + rules, no missing-tick check (clean)", "Proposed v2: OCR + rules, no missing-tick check (degraded)"
tests = {
    "v3 vs v2, clean": mcnemar(V3C, V2C),
    "v3 vs v2, degraded": mcnemar(V3D, V2D),
    "v3 clean vs oracle": mcnemar(V3C, "Oracle metadata + rules (upper bound)"),
    "v3 clean vs always-flag": mcnemar(V3C, "Baseline B0: always flag"),
}

diag = {}
for name, ocr, key in (("clean", oc, "file"), ("degraded", od, "file_degraded")):
    ok_read = [len(ocr[r[key]]["left"]) >= 3 for r in test]
    ok_type = [ocr[r[key]]["chart_type"] == r["chart_type"] for r in test]
    diag[name] = {"readable_rate": float(np.mean(ok_read)), "chart_type_acc": float(np.mean(ok_type))}

json.dump({"n_test_charts": len(test), "n_test_pairs": len(pairs), "summary": summary,
           "mcnemar": tests, "diagnostics": diag},
          open(f"{RES}/{out_name}.json", "w"), indent=1, default=float)
print("n test charts", len(test), "pairs", len(pairs))
for name, s in summary.items():
    print(f"{name:62s} TPR={s['TPR']:.3f} FPR={s['FPR']:.3f} bar={s['FPR_honest_bar']:.3f} "
          f"line={s['FPR_honest_line']:.3f} F1={s['F1']:.3f} p@5={s['precision_by_prevalence']['0.05']['mean']:.3f} "
          f"p@50={s['precision_by_prevalence']['0.5']['mean']:.3f} p@95={s['precision_by_prevalence']['0.95']['mean']:.3f} "
          f"abst_m={s['abstain_misleading']:.3f} abst_h={s['abstain_honest']:.3f}")
print(json.dumps(tests, indent=1)); print(json.dumps(diag, indent=1))
