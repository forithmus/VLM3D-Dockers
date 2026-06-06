#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute TP / FN / FP totals and CRG-style score:

    X  =  (#labels per exam) * (#exams)
    A  =  total positives in ground truth
    r  =  (X - A) / (2A)
    U  =  (X - A) / 2              (equivalent to A * r)
    s  =  r * TP - r * FN - FP
    CRG =  U / (2U - s)

Outputs a JSON file with all intermediate values.
"""

import argparse, json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import binarize

LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening"
]

def clean_accession(series):
    return (series.str.replace('.nii.gz',  '', regex=False)
            .str.replace('.npz',    '', regex=False)
            .str.replace('_embedded','', regex=False))

def main(pred_csv: Path, gt_csv: Path, out_json: Path):
    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(gt_csv)

    pred['AccessionNo'] = clean_accession(pred['AccessionNo'])
    gt  ['AccessionNo'] = clean_accession(gt['AccessionNo'])

    # remove any text columns that sneak in
    for df in (pred, gt):
        df.drop(columns=[c for c in df.columns
                         if c.lower().startswith("findings")], errors='ignore', inplace=True)

    pred.set_index('AccessionNo', inplace=True)
    gt.set_index('AccessionNo',   inplace=True)

    merged = pred.reindex(gt.index).astype(int)

    # --- Counters ------------------------------------------------------ #
    TP = ((merged[LABEL_COLS] == 1) & (gt[LABEL_COLS] == 1)).sum().sum()
    FN = ((merged[LABEL_COLS] == 0) & (gt[LABEL_COLS] == 1)).sum().sum()
    FP = ((merged[LABEL_COLS] == 1) & (gt[LABEL_COLS] == 0)).sum().sum()

    num_images = len(merged)
    X = len(LABEL_COLS) * num_images
    A = int(gt[LABEL_COLS].sum().sum())

    if A == 0:
        raise ValueError("Ground-truth contains zero positive labels; CRG undefined.")

    r = (X - A) / (2 * A)
    U = (X - A) / 2
    s = r * TP - r * FN - FP
    crg = U / (2 * U - s) if (2 * U - s) != 0 else 0

    result = {                   # always cast to plain Python scalars
        "TP":   int(TP),
        "FN":   int(FN),
        "FP":   int(FP),
        "X":    int(X),
        "A":    int(A),
        "r":    float(r),
        "U":    float(U),
        "score_s": float(s),
        "CRG":  float(crg)
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("CRG metrics →", out_json)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--gt_csv",   required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    main(Path(args.pred_csv), Path(args.gt_csv), Path(args.out_json))
