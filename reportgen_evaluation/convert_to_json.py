#!/usr/bin/env python3
"""
csv2generated_reports_json.py
=============================

• Reads   : /mnt/tiger/vlm3dchallenge/hidden_set_mapping.csv
• Writes  : generated_reports_hidden_set_2000.json
• Output schema (excerpt):
    {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": [
            {
                "input_image_name": "1.2.840.113704.1.111.10004.1691060324.6_B_Toraks.mha",
                "report": "<concat_text>"
            },
            …
        ],
        "version": { "major": 1, "minor": 0 }
    }
"""

import json
from pathlib import Path

import pandas as pd

CSV_PATH   = Path("/mnt/tiger/vlm3dchallenge/hidden_set_mapping.csv")
OUT_JSON   = Path("generated_reports_hidden_set_2000.json")
N_ROWS     = 2000   # take the first 2 000 rows

def main() -> None:
    # 1. Load CSV
    df = pd.read_csv(CSV_PATH, nrows=N_ROWS)

    # 2. Build records
    records = []
    for _, row in df.iterrows():
        nii_path  = Path(row["nii_path"])          # hidden_set/…/filename.nii.gz
        fname     = nii_path.name                  # filename.nii.gz
        mha_name  = fname.replace(".nii.gz", ".mha")
        report    = row["concat_text"]

        records.append({
            "input_image_name": mha_name,
            "report": report
        })

    # 3. Assemble wrapper dict
    wrapper = {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": records,
        "version": {"major": 1, "minor": 0}
    }

    # 4. Write JSON
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} entries → {OUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
