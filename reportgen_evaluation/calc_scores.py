#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute multi-label classification metrics.

Outputs `/output/classification_scores.json` with:

{
  "per_pathology": [
      {"name": "...", "precision": 0.88, "recall": 0.79, "f1": 0.83, "accuracy": 0.91},
      …
  ],
  "macro": {"precision": …, "recall": …, "f1": …, "accuracy": …}
}
"""

import argparse, json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate(pred_csv: Path, gt_csv: Path, out_json: Path):

    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(gt_csv)

    pred['AccessionNo'] = pred['AccessionNo'].str.replace('.npz',  '', regex=False)
    gt['AccessionNo']   = gt['AccessionNo'].str.replace('.nii.gz', '', regex=False)

    pred.set_index('AccessionNo', inplace=True)
    gt.set_index('AccessionNo',   inplace=True)

    pred = pred.reindex(gt.index).astype(int)

    results = {"per_pathology": []}
    precs, recs, f1s, accs = [], [], [], []

    for col in gt.columns:
        p = precision_score(gt[col], pred[col])
        r = recall_score(gt[col], pred[col])
        f = f1_score(gt[col],      pred[col])
        a = accuracy_score(gt[col], pred[col])

        results["per_pathology"].append(
            {"name": col, "precision": p, "recall": r, "f1": f, "accuracy": a}
        )
        precs.append(p); recs.append(r); f1s.append(f); accs.append(a)

    results["macro"] = {
        "precision": sum(precs) / len(precs),
        "recall":    sum(recs)  / len(recs),
        "f1":        sum(f1s)   / len(f1s),
        "accuracy":  sum(accs)  / len(accs),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Classification metrics →", out_json)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv",   required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    evaluate(Path(args.pred_csv), Path(args.gt_csv), Path(args.out_json))
