#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert inferred reports → multi-label pathology predictions (CSV).

Call from evaluate.py:
    python infer.py --input_json /input/inferred.json --model_path … --out_csv …
"""

import argparse, json, time
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler

from classifier import RadBertClassifier
from dataset import CTDataset
from model_trainer import ModelTrainer

# ------------------------------------------------------------------ #
def run_inference(json_path: Path, model_path: Path, out_csv: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    with open(json_path, "r", encoding="utf-8") as f:
        in_json = json.load(f)
    raw = in_json[0]["outputs"][0]["value"]

    items = raw["generated_reports"] if isinstance(raw, dict) else raw
    records = []
    for it in items:
        fname  = it["input_image_name"]
        report = it["report"]
        accession = Path(fname).stem
        records.append({"AccessionNo": accession, "reports": report})
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} reports from {json_path}")

    # ------------ labels placeholder ------------------------------------ #
    label_cols = [
        "Medical material", "Arterial wall calcification", "Cardiomegaly",
        "Pericardial effusion", "Coronary artery wall calcification",
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
        "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
        "Pleural effusion", "Mosaic attenuation pattern",
        "Peribronchial thickening", "Consolidation",
        "Bronchiectasis", "Interlobular septal thickening"
    ]
    for col in label_cols:
        df[col] = 0  # dummy

    # ------------ DataLoader -------------------------------------------- #
    ds = CTDataset(df, len(label_cols), label_cols, max_length=512, infer=True)
    loader = DataLoader(ds,
                        sampler=SequentialSampler(ds),
                        batch_size=32,
                        num_workers=4,
                        pin_memory=True)

    # ------------ Model -------------------------------------------------- #
    model = RadBertClassifier(n_classes=len(label_cols))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device).eval()

    trainer = ModelTrainer(model,
                           {"test": loader},
                           len(label_cols),
                           0,
                           AdamW(model.parameters(), lr=2e-5),
                           None,
                           device,
                           None,
                           label_cols)

    print("→  running inference")
    start = time.time()
    preds = trainer.infer()  # ndarray [N, C]
    print(f"Done in {time.time()-start:.1f}s")

    out = pd.DataFrame(preds, columns=label_cols)
    out.insert(0, "AccessionNo", df["AccessionNo"])
    out.to_csv(out_csv, index=False)
    print("Predictions saved to", out_csv)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv",    required=True)
    args = ap.parse_args()

    run_inference(Path(args.input_json), Path(args.model_path), Path(args.out_csv))
