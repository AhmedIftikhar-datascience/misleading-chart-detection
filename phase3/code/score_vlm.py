"""Score the VLM output file and compare it with the rule pipeline on the SAME charts.

Usage:
  python3 score_vlm.py --vlm ../results/vlm_clean.jsonl --rules-results ../results/ocr_clean.json \
      --manifest ../data/manifest.json --out ../results/vlm_summary_clean.json

Outputs a JSON summary (TPR, FPR overall and by honest class, F1, error count) for the VLM
and for the proposed rule pipeline, both restricted to the charts the VLM actually answered.
A McNemar test compares the two on those charts.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from detect import apply_rules  # noqa: E402


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if tp else 0.0
    return {"TPR": tpr, "FPR": fpr, "precision": prec, "F1": f1, "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--rules-results", required=True, help="OCR evidence cache, e.g. ocr_clean.json")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    manifest = {r["file"]: r for r in json.load(open(a.manifest))}
    manifest.update({r["file_degraded"]: r for r in manifest.values()})
    ocr = json.load(open(a.rules_results))
    recs = [json.loads(l) for l in open(a.vlm)]
    errors = sum(1 for r in recs if r["error"])
    ok = [r for r in recs if r["error"] is None]

    y = [r["label"] for r in ok]
    kinds = [r["kind"] for r in ok]
    vlm_pred = [r["pred"] for r in ok]
    rule_pred = []
    for r in ok:
        flagged, _, _ = apply_rules(ocr[r["file"]], missing_check=True)
        rule_pred.append(int(flagged))

    def by_kind(pred):
        out = {}
        for k in ("honest_bar", "honest_line"):
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            out[k + "_FPR"] = float(np.mean([pred[i] for i in idx])) if idx else None
        return out

    def recall_by_type(pred):
        out = {}
        for k in ("truncated", "inverted", "dual"):
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            out[k] = float(np.mean([pred[i] for i in idx])) if idx else None
        return out

    vlm_m = metrics(y, vlm_pred); rule_m = metrics(y, rule_pred)
    vlm_m.update(by_kind(vlm_pred)); rule_m.update(by_kind(rule_pred))
    vlm_m["recall_by_type"] = recall_by_type(vlm_pred)
    rule_m["recall_by_type"] = recall_by_type(rule_pred)

    # McNemar on correctness, same charts
    cv = np.array([p == t for p, t in zip(vlm_pred, y)])
    cr = np.array([p == t for p, t in zip(rule_pred, y)])
    b01 = int((cv & ~cr).sum()); b10 = int((~cv & cr).sum())
    if b01 + b10:
        chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        pval = float(1 - stats.chi2.cdf(chi, 1))
    else:
        pval = 1.0

    # what the VLM said about the misleader type (for misleading charts only)
    type_match = {}
    for t in ("truncated", "inverted", "dual"):
        idx = [i for i, r in enumerate(ok) if r["kind"] == t]
        type_match[t] = float(np.mean([ok[i]["pred_misleader"] == t for i in idx])) if idx else None

    summary = {
        "model": ok[0]["model"] if ok else None,
        "condition": ok[0]["condition"] if ok else None,
        "charts_requested": len(recs), "charts_answered": len(ok), "api_errors": errors,
        "charts_per_kind": {k: sum(1 for r in ok if r["kind"] == k)
                            for k in ("honest_bar", "honest_line", "truncated", "inverted", "dual")},
        "VLM": vlm_m, "rules_same_charts": rule_m,
        "mcnemar_vlm_vs_rules": {"vlm_right_rules_wrong": b01, "rules_right_vlm_wrong": b10, "p": pval},
        "vlm_misleader_type_accuracy": type_match,
    }
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
