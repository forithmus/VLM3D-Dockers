#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for the MR-RATE abnormality-classification track.
Forithmus mounts BOTH participant predictions and host ground truth from /input/:
/input/
  predictions/   - first *.json here is the participant submission
  ground_truth/  - ground_truth.csv
/output/
  metrics.json   - classification scores (precision/recall/F1/accuracy/AUROC, macro + per-label)
"""
import json, subprocess, sys
from pathlib import Path

INPUT_DIR  = Path("/input/predictions")
GT_DIR     = Path("/input/ground_truth")
OUTPUT_DIR = Path("/output")
CODE_DIR   = Path("/opt/app")

JSON2CSV_SCRIPT = CODE_DIR / "json_to_csv.py"
CLS_SCRIPT      = CODE_DIR / "calc_scores.py"

GT_CSV     = GT_DIR / "ground_truth.csv"
CSV_BIN    = OUTPUT_DIR / "predictions_bin.csv"
CSV_PROB   = OUTPUT_DIR / "predictions_prob.csv"
FINAL_JSON = OUTPUT_DIR / "metrics.json"

def _run(script: Path, *args):
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def _first_json(dir_: Path) -> Path:
    files = sorted(dir_.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json submission found in {dir_}/")
    return files[0]

def main(thresh: float = 0.5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_json = _first_json(INPUT_DIR)
    print(f"Using submission JSON: {pred_json}")
    if not GT_CSV.exists():
        raise FileNotFoundError(f"Ground-truth CSV missing at {GT_CSV}")

    _run(JSON2CSV_SCRIPT,
         "--pred_json",    pred_json,
         "--out_csv_bin",  CSV_BIN,
         "--out_csv_prob", CSV_PROB,
         "--thresh",       str(thresh))

    _run(CLS_SCRIPT,
         "--pred_csv",  CSV_BIN,
         "--gt_csv",    GT_CSV,
         "--pred_json", pred_json,
         "--out_json",  FINAL_JSON)

    print("All metrics written to", FINAL_JSON)

if __name__ == "__main__":
    main(thresh=0.5)
