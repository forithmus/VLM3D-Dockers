#!/usr/bin/env python3
"""
MR-RATE MIL classification baseline — Forithmus submission entrypoint.

Reads challenge cases from /input (one folder per study containing atlas/
volumes produced by the platform's merged input format), encodes each study
with the frozen MR-RATE visual encoder (V-JEPA backbone + LoRA), runs the
frozen 74-class MIL head, and writes per-study pathology probabilities to
/output/predictions.json in the schema the evaluation container parses.

Fully offline: every artifact (V-JEPA backbone, encoder+MIL checkpoints,
CXR-BERT snapshot, vjepa2 hub code) comes from /weights; HF_HUB_OFFLINE and a
pre-seeded TORCH_HOME prevent any network access.

Volume preprocessing mirrors MRReportDatasetInfer exactly (same resample /
minmax normalise / crop_or_pad, same atlas_space contract), so the encoder
sees distributionally matched inputs.
"""
import json
import os
import sys
import time
from pathlib import Path

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
WEIGHTS = Path(os.environ.get("FORITHMUS_WEIGHTS_DIR", "/weights"))
BUNDLE = WEIGHTS / "bundle"
UPSTREAM = Path("/opt/MR-RATE/contrastive-pretraining")

# Offline guards — must be set before torch/transformers import anything.
os.environ.setdefault("HF_HOME", str(WEIGHTS / "hf"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TORCH_HOME", "/opt/torchhub")

sys.path.insert(0, str(BUNDLE / "scripts"))
sys.path.insert(0, str(UPSTREAM / "scripts"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, "/opt/MR-RATE/report-generation-training/src")

import numpy as np
import torch

from data import NORMALIZERS  # upstream
from run_inference_common import (
    discover_studies, load_study_stack, evenly_spaced_indices,
    MAX_TOKENS_PER_STUDY,
)
from load_encoder import load_encoder                              # bundle
from mrrate_report_training.mil import load_frozen_mil, infer_mil  # writer repo

CONFIG = Path(os.environ.get("ENCODER_CONFIG", "/tmp/encoder.local.yaml"))
MIL_CKPT = BUNDLE / "weights" / "mil" / "mil_head_mr11000_s44.pt"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[baseline] device={device} dtype={dtype}", flush=True)

    studies = discover_studies(INPUT_DIR)
    if not studies:
        print(f"ERROR: no studies with atlas volumes under {INPUT_DIR}", file=sys.stderr)
        return 1
    print(f"[baseline] {len(studies)} studies discovered", flush=True)

    encoder, _dim = load_encoder(str(CONFIG), task="tokens", upstream_root=str(UPSTREAM))
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    encoder.requires_grad_(False)

    head, label_names, _thresholds = load_frozen_mil(
        MIL_CKPT, UPSTREAM, expected_dim=512,
    )
    head.to(device=device)
    print(f"[baseline] encoder + MIL head loaded ({len(label_names)} classes)", flush=True)

    normalizer = NORMALIZERS["minmax"]()
    predictions = []
    t0 = time.time()
    for i, (study, paths) in enumerate(studies, 1):
        try:
            stack = load_study_stack(paths, normalizer)          # [N,1,D,H,W]
            images = stack.unsqueeze(0).to(device=device, dtype=dtype)  # [1,N,1,D,H,W]
            mask = torch.ones((1, stack.shape[0]), dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                visual_tokens, token_mask = encoder(
                    text_input=None, image=images, device=device,
                    real_volume_mask=mask, return_loss=False,
                    return_visual_tokens=True,
                )
            tokens = visual_tokens[0, token_mask[0].bool()].detach()
            sel = evenly_spaced_indices(tokens.shape[0], MAX_TOKENS_PER_STUDY)
            if sel.size != tokens.shape[0]:
                tokens = tokens.index_select(0, torch.from_numpy(sel).to(tokens.device))
            logits = infer_mil(head, tokens.float())
            probs = torch.sigmoid(logits).flatten().cpu().numpy()
            predictions.append({
                "input_image_name": f"{study}.nii.gz",
                "probabilities": {n: float(p) for n, p in zip(label_names, probs)},
            })
            del stack, images, tokens
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            # A failed study still gets a row (all zeros) so scoring can proceed.
            print(f"WARN {study}: {type(e).__name__} {e}", file=sys.stderr, flush=True)
            predictions.append({
                "input_image_name": f"{study}.nii.gz",
                "probabilities": {n: 0.0 for n in label_names},
            })
        if i % 10 == 0 or i == len(studies):
            el = time.time() - t0
            print(f"[baseline] {i}/{len(studies)}  {el/60:.1f} min "
                  f"({el/i:.1f}s/study)", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = [{"outputs": [{"value": {"predictions": predictions}}]}]
    with open(OUTPUT_DIR / "predictions.json", "w") as f:
        json.dump(doc, f)
    print(f"[baseline] wrote {OUTPUT_DIR/'predictions.json'} "
          f"({len(predictions)} studies)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
