#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for the VLM3D report-generation track, ported to
the Forithmus eval-container contract.

Forithmus mounts BOTH participant predictions and host ground truth from
/input/ at runtime. Ground truth is NOT baked into the image (unlike the
grand-challenge.org image this was derived from).

/input/
  predictions/   - first *.json here is the participant submission
  ground_truth/  - ground_truth.json + ground_truth.csv
/output/
  metrics.json   - merged scores (generation + classification + crg)
"""

import json, subprocess, sys
from pathlib import Path

INPUT_DIR   = Path("/input/predictions")
GT_DIR      = Path("/input/ground_truth")
OUTPUT_DIR  = Path("/output")
CODE_DIR    = Path("/opt/app")

INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT   = CODE_DIR / "calc_scores.py"
CRG_SCRIPT   = CODE_DIR / "crg_score.py"
NLG_SCRIPT   = CODE_DIR / "nlg_metrics.py"

GT_JSON = GT_DIR / "ground_truth.json"
GT_CSV  = GT_DIR / "ground_truth.csv"

CSV_PRED  = OUTPUT_DIR / "inferred.csv"
CLS_JSON  = OUTPUT_DIR / "classification_scores.json"
CRG_JSON  = OUTPUT_DIR / "crg_scores.json"
NLG_JSON  = OUTPUT_DIR / "nlg_scores.json"
FINAL_JSON = OUTPUT_DIR / "metrics.json"


def find_pred_json() -> Path:
    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json submission found in {INPUT_DIR}/")
    return files[0]


def run(script: Path, *args):
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    pred_json = find_pred_json()
    print("Using prediction file:", pred_json)

    # 1. inference → CSV
    run(INFER_SCRIPT,
        "--input_json", pred_json,
        "--model_path", CODE_DIR / "RadBertClassifier.pth",
        "--out_csv",    CSV_PRED)

    # 2. multi-label classification scores
    run(CLS_SCRIPT,
        "--pred_csv", CSV_PRED,
        "--gt_csv",   GT_CSV,
        "--out_json", CLS_JSON)

    # 3. CRG metrics
    run(CRG_SCRIPT,
        "--pred_csv", CSV_PRED,
        "--gt_csv",   GT_CSV,
        "--out_json", CRG_JSON)

    # 4. NLG metrics
    run(NLG_SCRIPT,
        "--pred_json", pred_json,
        "--gt_json",   GT_JSON,
        "--out_json",  NLG_JSON)

    combined = {
        "generation":     load(NLG_JSON),
        "classification": load(CLS_JSON),
        "crg":            load(CRG_JSON)
    }
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print("\n✅  All metrics written to", FINAL_JSON)


if __name__ == "__main__":
    main()
