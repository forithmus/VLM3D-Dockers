#!/usr/bin/env python3
"""
Binary multi-label metrics for the MR report-generation track.

Both prediction and ground truth are 0/1 CSVs over the same 32-label subset
(the LLM extractor emits binary calls, so unlike the image-classification
track there are no probabilities and hence no AUROC).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def main(pred_csv: Path, gt_csv: Path, out_json: Path):
    pred = pd.read_csv(pred_csv).set_index("AccessionNo")
    gt = pd.read_csv(gt_csv).set_index("AccessionNo")
    labels = [c for c in gt.columns]
    pred = pred.reindex(gt.index).fillna(0).astype(int)

    results = {"per_pathology": []}
    p_all, r_all, f_all, a_all = [], [], [], []
    for col in labels:
        y, yhat = gt[col].to_numpy(int), pred[col].to_numpy(int)
        p = precision_score(y, yhat, zero_division=0)
        r = recall_score(y, yhat, zero_division=0)
        f = f1_score(y, yhat, zero_division=0)
        a = accuracy_score(y, yhat)
        results["per_pathology"].append(
            {"name": col, "precision": p, "recall": r, "f1": f, "accuracy": a})
        p_all.append(p); r_all.append(r); f_all.append(f); a_all.append(a)
    results["macro"] = {
        "precision": float(np.mean(p_all)), "recall": float(np.mean(r_all)),
        "f1": float(np.mean(f_all)), "accuracy": float(np.mean(a_all)),
    }
    out_json.write_text(json.dumps(results, indent=2))
    print("classification metrics ->", out_json)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, type=Path)
    ap.add_argument("--gt_csv", required=True, type=Path)
    ap.add_argument("--out_json", required=True, type=Path)
    a = ap.parse_args()
    main(a.pred_csv, a.gt_csv, a.out_json)
