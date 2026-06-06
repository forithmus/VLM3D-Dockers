#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert submission JSON (18 abnormalities) into *two* CSV files:
  • binary 0/1 labels            → --out_csv_bin
  • raw probability scores       → --out_csv_prob

Example
-------
python json_to_csv.py \
    --pred_json    /input/predictions.json \
    --out_csv_bin  /output/predictions_bin.csv \
    --out_csv_prob /output/predictions_prob.csv \
    --thresh       0.5
"""

import argparse, json
from pathlib import Path

import pandas as pd

LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]

def main(pred_json: Path, out_bin: Path, out_prob: Path, thresh: float):
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data[0]["outputs"][0]["value"]["predictions"]
    rows_prob, rows_bin = [], []

    for item in preds:
        acc = Path(item["input_image_name"]).stem
        scores = {lab: float(item["probabilities"].get(lab, 0.0)) for lab in LABEL_COLS}

        rows_prob.append({"AccessionNo": acc.replace(".nii",""), **scores})
        rows_bin.append({"AccessionNo": acc.replace(".nii",""),
                          **{lab: int(v >= thresh) for lab, v in scores.items()}})

    pd.DataFrame(rows_prob).to_csv(out_prob, index=False)
    pd.DataFrame(rows_bin ).to_csv(out_bin , index=False)
    print(pd.DataFrame(rows_prob ))
    print("testo")
    print("↳ wrote:", out_prob)
    print("↳ wrote:", out_bin)
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json",    required=True, type=Path)
    ap.add_argument("--out_csv_bin",  required=True, type=Path)
    ap.add_argument("--out_csv_prob", required=True, type=Path)
    ap.add_argument("--thresh",       type=float, default=0.5,
                    help="Probability ≥ THRESH → positive label in binary CSV")
    args = ap.parse_args()
    main(args.pred_json, args.out_csv_bin, args.out_csv_prob, args.thresh)
