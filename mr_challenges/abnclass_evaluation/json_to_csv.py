#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert submission JSON (37 MR-RATE abnormalities) into two CSVs:
  • binary 0/1 labels           -> --out_csv_bin
  • raw probability scores      -> --out_csv_prob
"""
import argparse, json
from pathlib import Path
import pandas as pd
from labels import LABEL_COLS

def main(pred_json: Path, out_bin: Path, out_prob: Path, thresh: float):
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data[0]["outputs"][0]["value"]["predictions"]
    rows_prob, rows_bin = [], []

    for item in preds:
        acc = Path(item["input_image_name"]).stem
        scores = {lab: float(item["probabilities"].get(lab, 0.0)) for lab in LABEL_COLS}

        rows_prob.append({"AccessionNo": acc.replace(".nii", ""), **scores})
        rows_bin.append({"AccessionNo": acc.replace(".nii", ""),
                          **{lab: int(v >= thresh) for lab, v in scores.items()}})

    pd.DataFrame(rows_prob).to_csv(out_prob, index=False)
    pd.DataFrame(rows_bin).to_csv(out_bin, index=False)
    print("↳ wrote:", out_prob)
    print("↳ wrote:", out_bin)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json",    required=True, type=Path)
    ap.add_argument("--out_csv_bin",  required=True, type=Path)
    ap.add_argument("--out_csv_prob", required=True, type=Path)
    ap.add_argument("--thresh",       type=float, default=0.5)
    args = ap.parse_args()
    main(args.pred_json, args.out_csv_bin, args.out_csv_prob, args.thresh)
