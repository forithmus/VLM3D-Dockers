#!/usr/bin/env python3
"""
MR-RATE report-generation baseline — Forithmus submission entrypoint.

Encoder + MIL conditioning + MedGemma-4B LoRA writer, exactly the bundle's
inference pipeline, driven by challenge inputs instead of the training-repo
dataset (no report targets or split membership exist at submission time).

Writes /output/predictions.json:
  {"generated_reports": [{"input_image_name": "<STUDY>.nii.gz",
                          "report": "<generated findings>"}]}
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

os.environ.setdefault("HF_HOME", str(WEIGHTS / "hf"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TORCH_HOME", "/opt/torchhub")

sys.path.insert(0, str(BUNDLE / "scripts"))
sys.path.insert(0, str(UPSTREAM / "scripts"))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, "/opt/MR-RATE/report-generation-training/src")

import torch

from run_inference_common import (  # shared with the classification baseline
    discover_studies, load_study_stack, evenly_spaced_indices,
    TARGET_SPACING, MAX_TOKENS_PER_STUDY,
)
from load_encoder import load_encoder
from mrrate_report_training.mil import load_frozen_mil, infer_mil
from mrrate_report_training.model import (
    ReportWriter, build_gemma_writer, label_semantic_embeddings,
)
from mrrate_report_training.generate import load_writer_checkpoint

CONFIG = Path(os.environ.get("ENCODER_CONFIG", "/tmp/encoder.local.yaml"))
MIL_CKPT = BUNDLE / "weights" / "mil" / "mil_head_mr11000_s44.pt"
WRITER_CKPT = BUNDLE / "weights" / "writer" / "last.pt"
LLM_PATH = WEIGHTS / "medgemma-4b-it"

# writer hyperparameters from config_mr11000_medgemma4b.yaml
WRITER = dict(num_visual_queries=512, resampler_depth=2, resampler_heads=8,
              lora_r=16, lora_alpha=32, max_target_tokens=1536,
              mil_conditioning="all_classes")

from data import NORMALIZERS  # noqa: E402  (after sys.path setup)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        return 1
    device = torch.device("cuda", 0)

    studies = discover_studies(INPUT_DIR)
    if not studies:
        print(f"ERROR: no studies under {INPUT_DIR}", file=sys.stderr)
        return 1
    print(f"[reportgen] {len(studies)} studies", flush=True)

    encoder, _dim = load_encoder(str(CONFIG), task="tokens",
                                 upstream_root=str(UPSTREAM))
    encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    mil_head, label_names, thresholds = load_frozen_mil(
        MIL_CKPT, UPSTREAM, expected_dim=512)
    mil_head.to(device).eval()
    thresholds = thresholds.to(device)

    llm, tokenizer, hidden = build_gemma_writer(
        str(LLM_PATH), device,
        lora_r=WRITER["lora_r"], lora_alpha=WRITER["lora_alpha"])
    semantics = label_semantic_embeddings(llm, tokenizer, label_names)
    model = ReportWriter(
        llm, tokenizer, semantics, visual_dim=512,
        num_visual_queries=WRITER["num_visual_queries"],
        resampler_depth=WRITER["resampler_depth"],
        resampler_heads=WRITER["resampler_heads"],
        max_target_tokens=WRITER["max_target_tokens"],
        mil_conditioning=WRITER["mil_conditioning"],
    ).to(device)
    load_writer_checkpoint(WRITER_CKPT, model, label_names)
    model.eval()
    print("[reportgen] writer loaded", flush=True)

    normalizer = NORMALIZERS["minmax"]()
    results = []
    t0 = time.time()
    for i, (study, paths) in enumerate(studies, 1):
        try:
            stack = load_study_stack(paths, normalizer)
            images = stack.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            mask = torch.ones((1, stack.shape[0]), dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                visual_tokens, token_mask = encoder(
                    text_input=None, image=images, device=device,
                    real_volume_mask=mask, return_loss=False,
                    return_visual_tokens=True)
            tokens = visual_tokens[0, token_mask[0].bool()].detach()
            sel = evenly_spaced_indices(tokens.shape[0], MAX_TOKENS_PER_STUDY)
            if sel.size != tokens.shape[0]:
                tokens = tokens.index_select(
                    0, torch.from_numpy(sel).to(tokens.device))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mil_logits = infer_mil(mil_head, tokens)
                text = model.generate(
                    tokens, mil_logits, thresholds,
                    max_new_tokens=WRITER["max_target_tokens"])
            results.append({"input_image_name": f"{study}.nii.gz",
                            "report": text})
            del stack, images, tokens
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"WARN {study}: {type(e).__name__} {e}", file=sys.stderr, flush=True)
            results.append({"input_image_name": f"{study}.nii.gz", "report": ""})
        if i % 5 == 0 or i == len(studies):
            el = time.time() - t0
            print(f"[reportgen] {i}/{len(studies)}  {el/60:.1f} min "
                  f"({el/i:.1f}s/study)", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "predictions.json", "w") as f:
        json.dump({"generated_reports": results}, f)
    print(f"[reportgen] wrote predictions.json ({len(results)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
