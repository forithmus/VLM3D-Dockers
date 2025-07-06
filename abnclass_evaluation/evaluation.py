#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for the Abnormality-Classification Challenge.

It performs the following steps:

1. Locate the participant’s submission JSON in /input
2. Convert that JSON → (a) binary-label CSV   (b) probability CSV
3. Run multi-label metrics (P/R/F1/Accuracy + AUROC) on the binary CSV
4. Run CRG metrics on the probability CSV
5. Merge results into /output/metrics.json   (consumed by the leaderboard)

Directory layout (inside the Docker container)
----------------------------------------------
/input                 ─ participant submission (exactly ONE *.json)
/output                ─ metrics written here
/opt/app               ─ this file + helper scripts
└─ ground-truth
     └─ ground_truth.csv
"""

import json
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Fixed paths and helper script names                                        #
# --------------------------------------------------------------------------- #
INPUT_DIR   = Path("/input")
OUTPUT_DIR  = Path("/output")
CODE_DIR    = Path("/opt/app")
GT_DIR      = CODE_DIR / "ground-truth"

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
def _run(script: Path, *args: Path | str):
    """Echo + run a helper script via the same Python interpreter."""
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _first_json(dir_: Path) -> Path:
    """Return the first *.json in a directory; raise if none exist."""
    files = sorted(dir_.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json found in {dir_}")
    return files[0]


def _load_json(p: Path):
    with p.open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
def main(thresh: float = 0.5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 0️⃣  Submission file                                               #
    # ------------------------------------------------------------------ #
    pred_json = _first_json(INPUT_DIR)
    print(f"Using submission JSON: {pred_json}")

    # ------------------------------------------------------------------ #
    # 1️⃣  JSON → CSV (binary + probabilities)                           #
    # ------------------------------------------------------------------ #
    _run(JSON2CSV_SCRIPT,
         "--pred_json",    pred_json,
         "--out_csv_bin",  CSV_BIN,
         "--out_csv_prob", CSV_PROB,
         "--thresh",       str(thresh))

    # ------------------------------------------------------------------ #
    # 2️⃣  Classification metrics (binary CSV + probs from JSON)         #
    # ------------------------------------------------------------------ #
    _run(CLS_SCRIPT,
         "--pred_csv",  CSV_BIN,
         "--gt_csv",    GT_CSV,
         "--pred_json", pred_json,      # gives AUROC its probabilities
         "--out_json",  CLS_JSON)

    # ------------------------------------------------------------------ #
    # 3️⃣  CRG metrics (probability CSV)                                 #
    # ------------------------------------------------------------------ #
    _run(CRG_SCRIPT,
         "--pred_csv", CSV_BIN,
         "--gt_csv",   GT_CSV,
         "--out_json", CRG_JSON)

    # ------------------------------------------------------------------ #
    # 4️⃣  Combine and save                                              #
    # ------------------------------------------------------------------ #
    combined = {
        "classification": _load_json(CLS_JSON),
        "crg":            _load_json(CRG_JSON),
    }

    FINAL_JSON.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("\n✅  All metrics written to", FINAL_JSON)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # If you ever want a different threshold (e.g. 0.4), change it here.
    main(thresh=0.5)
