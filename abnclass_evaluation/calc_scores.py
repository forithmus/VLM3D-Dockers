#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-label metrics for the Abnormality-Classification Challenge
────────────────────────────────────────────────────────────────
Outputs precision, recall, F1, accuracy **and AUROC**
  • per pathology
  • macro-averaged
"""

import argparse, json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)

LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]


# --------------------------------------------------------------------------- #
def _json_probs_to_df(json_path: Path) -> pd.DataFrame:
    """Extract probability scores (0-1) from submission JSON → DataFrame."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data[0]["outputs"][0]["value"]["predictions"]:
        acc = Path(item["input_image_name"]).stem
        scores = {lab: float(item["probabilities"].get(lab, 0.0)) for lab in LABEL_COLS}
        rows.append({"AccessionNo": acc.replace(".nii",""), **scores})

    return pd.DataFrame(rows).set_index("AccessionNo")


# --------------------------------------------------------------------------- #
def evaluate(pred_csv: Path, gt_csv: Path, out_json: Path,
             prob_json: Optional[Path] = None) -> None:

    pred_df = pd.read_csv(pred_csv).set_index("AccessionNo").astype(int)
    gt_df   = pd.read_csv(gt_csv).set_index("AccessionNo").astype(int)

    # Align indices (raises if mismatch)
    pred_df = pred_df.reindex(gt_df.index)

    # Load probabilities for AUROC
    prob_arr = None

    if prob_json:
        prob_df = _json_probs_to_df(prob_json).reindex(gt_df.index)
        prob_arr = prob_df[LABEL_COLS].to_numpy(dtype=float)
    y_true = gt_df[LABEL_COLS].to_numpy(dtype=int)
    y_pred = pred_df[LABEL_COLS].to_numpy(dtype=int)

    results = {"per_pathology": []}
    p_all, r_all, f_all, a_all, auc_all = [], [], [], [], []

    for i, col in enumerate(LABEL_COLS):
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score   (y_true[:, i], y_pred[:, i], zero_division=0)
        f = f1_score       (y_true[:, i], y_pred[:, i], zero_division=0)
        a = accuracy_score (y_true[:, i], y_pred[:, i])

        auc = None
        if prob_arr is not None:
            try:
                auc = roc_auc_score(y_true[:, i], prob_arr[:, i])
            except ValueError:  # only one class present
                pass

        results["per_pathology"].append(
            {"name": col, "precision": p, "recall": r,
             "f1": f, "accuracy": a, "auroc": auc}
        )
        p_all.append(p); r_all.append(r); f_all.append(f); a_all.append(a)
        auc_all.append(auc if auc is not None else np.nan)

    results["macro"] = {
        "precision": float(np.mean(p_all)),
        "recall":    float(np.mean(r_all)),
        "f1":        float(np.mean(f_all)),
        "accuracy":  float(np.mean(a_all)),
        "auroc":     float(np.nanmean(auc_all)) if prob_arr is not None else None,
    }

    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("✔ classification_scores.json →", out_json)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv",  required=True, type=Path)
    ap.add_argument("--gt_csv",    required=True, type=Path)
    ap.add_argument("--out_json",  required=True, type=Path)
    ap.add_argument("--pred_json", type=Path, default=None,
                    help="Submission JSON with probabilities (for AUROC)")
    args = ap.parse_args()
    evaluate(args.pred_csv, args.gt_csv, args.out_json, args.pred_json)
