#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for the VLM3D abnormality-classification track,
ported to the Forithmus eval-container contract.

Forithmus mounts BOTH participant predictions and host ground truth from
/input/ at runtime. Ground truth is NOT baked into the image (unlike the
grand-challenge.org image this was derived from).

/input/
  predictions/   - first *.json here is the participant submission
  ground_truth/  - ground_truth.csv
/output/
  metrics.json   - merged scores (classification + crg)
"""

import json
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Fixed paths and helper script names                                        #
# --------------------------------------------------------------------------- #
INPUT_DIR   = Path("/input/predictions")
GT_DIR      = Path("/input/ground_truth")
OUTPUT_DIR  = Path("/output")
CODE_DIR    = Path("/opt/app")

JSON2CSV_SCRIPT = CODE_DIR / "json_to_csv.py"     # produces bin+prob CSVs
CLS_SCRIPT      = CODE_DIR / "calc_scores.py"     # P/R/F1/Acc + AUROC
CRG_SCRIPT      = CODE_DIR / "crg_score.py"       # CRG score (needs probs)

#  Ground-truth and artefact paths
GT_CSV      = GT_DIR / "ground_truth.csv"

CSV_BIN     = OUTPUT_DIR / "predictions_bin.csv"
CSV_PROB    = OUTPUT_DIR / "predictions_prob.csv"
CLS_JSON    = OUTPUT_DIR / "classification_scores.json"
CRG_JSON    = OUTPUT_DIR / "crg_scores.json"
FINAL_JSON  = OUTPUT_DIR / "metrics.json"


# --------------------------------------------------------------------------- #
def _run(script: Path, *args):
    """Echo + run a helper script via the same Python interpreter."""
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _first_json(dir_: Path) -> Path:
    """Return the first *.json in a directory; raise if none exist."""
    files = sorted(dir_.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json submission found in {dir_}/")
    return files[0]


def _load_json(p: Path):
    with p.open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def main(thresh: float = 0.5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0  Submission file
    pred_json = _first_json(INPUT_DIR)
    print(f"Using submission JSON: {pred_json}")
    if not GT_CSV.exists():
        raise FileNotFoundError(f"Ground-truth CSV missing at {GT_CSV}")

    # 1  JSON -> CSV (binary + probabilities)
    _run(JSON2CSV_SCRIPT,
         "--pred_json",    pred_json,
         "--out_csv_bin",  CSV_BIN,
         "--out_csv_prob", CSV_PROB,
         "--thresh",       str(thresh))

    # 2  Classification metrics (binary CSV + probs from JSON)
    _run(CLS_SCRIPT,
         "--pred_csv",  CSV_BIN,
         "--gt_csv",    GT_CSV,
         "--pred_json", pred_json,
         "--out_json",  CLS_JSON)

    # 3  CRG metrics (probability CSV)
    _run(CRG_SCRIPT,
         "--pred_csv", CSV_BIN,
         "--gt_csv",   GT_CSV,
         "--out_json", CRG_JSON)

    # 4  Combine and save
    combined = {
        "classification": _load_json(CLS_JSON),
        "crg":            _load_json(CRG_JSON),
    }
    FINAL_JSON.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("All metrics written to", FINAL_JSON)


if __name__ == "__main__":
    main(thresh=0.5)
