#!/usr/bin/env python
"""
BTB3D submission orchestrator.

Reads CT volumes from /input/, encodes via MAGViT-2 (in-process), generates
radiology reports via LoRA-finetuned Llama-3.1-8B (4/8-bit quantized for T4),
writes /output/predictions.json in the Forithmus VLM3D challenge schema.

Idempotent: writes partial state to /checkpoint/state.json after every volume.
On SIGTERM, flushes state and exits 0 within the platform's 30-s grace window.

Environment variables (all optional):
  INPUT_DIR     default /input
  OUTPUT_DIR    default /output
  CHECKPOINT_DIR default /checkpoint
  WEIGHTS_DIR   default /opt/btb3d/weights
  BTB3D_QUANT   one of {4bit, 8bit, bf16}; default 4bit
  MAX_NEW_TOKENS default 512
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# ---------- configurable paths ----------
INPUT_DIR      = Path(os.environ.get("INPUT_DIR",      "/input"))
OUTPUT_DIR     = Path(os.environ.get("OUTPUT_DIR",     "/output"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoint"))
WEIGHTS_DIR    = Path(os.environ.get("WEIGHTS_DIR",    "/opt/btb3d/weights"))
QUANT          = os.environ.get("BTB3D_QUANT",         "4bit").lower()
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS",  "512"))

STATE_PATH      = CHECKPOINT_DIR / "state.json"
PREDICTIONS_OUT = OUTPUT_DIR / "predictions.json"

# Target resampling (matches CT-RATE preprocessing the model was trained on)
TARGET_SPACING = (1.5, 0.75, 0.75)   # (Z, X, Y) mm
HU_MIN, HU_MAX = -1000.0, 1000.0

# Fixed spatial size the encoder was trained on.
TARGET_SHAPE = (512, 512, 241)        # (X, Y, Z)
PAD_VALUE    = -1.0                    # post-normalization air

# Prompt the LoRA was trained on
PROMPT = "<image>\n<report_generation>"


# ---------- signal handling ----------
_stop_requested = False
def _on_sigterm(signum, frame):
    global _stop_requested
    print(f"[predict] SIGTERM received, will flush state and exit after current volume", file=sys.stderr, flush=True)
    _stop_requested = True
signal.signal(signal.SIGTERM, _on_sigterm)


# ---------- filename → input_image_name ----------
# Pattern: {batch}_{patient}_{series}.nii.gz
# input_image_name = series UUID (third segment, no extension)
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

def derive_image_name(nii_path: Path) -> str:
    stem = nii_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    parts = stem.split("_")
    # Prefer the last UUID-looking segment; fall back to the whole stem.
    for p in reversed(parts):
        if UUID_RE.fullmatch(p):
            return p
    return stem


# ---------- preprocessing (header-based; no CT-RATE CSV) ----------
def _center_crop_or_pad(t: torch.Tensor, target_xyz) -> torch.Tensor:
    """Mirror read_nii_preprocess in run_whole_dataset.py:96-127.
    Input/output layout: (X, Y, Z). Pads with PAD_VALUE."""
    dh, dw, dd = target_xyz
    h, w, d = t.shape
    h_start = max((h - dh) // 2, 0); h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0); w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0); d_end = min(d_start + dd, d)
    t = t[h_start:h_end, w_start:w_end, d_start:d_end]
    pad = (
        (dd - t.size(2)) // 2, dd - t.size(2) - (dd - t.size(2)) // 2,
        (dw - t.size(1)) // 2, dw - t.size(1) - (dw - t.size(1)) // 2,
        (dh - t.size(0)) // 2, dh - t.size(0) - (dh - t.size(0)) // 2,
    )
    return F.pad(t, pad, value=PAD_VALUE)


def _center_crop_axis2(t: torch.Tensor) -> torch.Tensor:
    """Mirror run_whole_dataset.py:246-261. Trim axis 2 to 1+4n length."""
    D = t.shape[2]
    n = (D - 1) // 4
    new_size = 1 + 4 * n
    start = (D - new_size) // 2
    slices = [slice(None)] * t.ndim
    slices[2] = slice(start, start + new_size)
    return t[tuple(slices)]


def load_and_preprocess(nii_path: Path) -> torch.Tensor:
    """Mirror read_nii_preprocess (production-data flavour).

    Production CT data is already in HU and has DICOM spacing in the NIfTI
    header, so no metadata CSV is needed (slope=1, intercept=0).

    Returns a (1, 1, Z=241, X=512, Y=512) float32 tensor in [-1, 1], ready for
    the encoder.
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.float32)             # (X, Y, Z), HU
    sx, sy, sz = (float(z) for z in img.header.get_zooms()[:3])

    # 1) Spacing resample: (X, Y, Z) -> (Z, X, Y) for trilinear, then back.
    data_zxy = np.transpose(data, (2, 0, 1))              # (Z, X, Y)
    t = torch.from_numpy(data_zxy).unsqueeze(0).unsqueeze(0)  # (1, 1, Z, X, Y)
    current = (sz, sx, sy)
    factors = [c / tgt for c, tgt in zip(current, TARGET_SPACING)]
    new_shape = [int(round(t.shape[2 + i] * factors[i])) for i in range(3)]
    t = F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)
    t = t[0, 0].permute(1, 2, 0).contiguous()             # → (X, Y, Z)

    # 2) HU clip + normalize to [-1, 1].
    t = t.clamp(HU_MIN, HU_MAX) / 1000.0

    # 3) Center crop/pad to (512, 512, 241).
    t = _center_crop_or_pad(t, TARGET_SHAPE)

    # 4) Permute (X, Y, Z) -> (Z, X, Y) and add (batch, channel).
    t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)      # (1, 1, 241, 512, 512)
    return t.contiguous()


# ---------- MAGViT-2 encoder ----------
def load_tokenizer():
    """Build VisionTokenizer the way run_whole_dataset.py:276-286 does."""
    sys.path.insert(0, "/opt/btb3d/encoder-decoder")
    from modeling.magvit_model import VisionTokenizer
    from src.utils import get_config

    cfg = get_config("/opt/btb3d/encoder-decoder/configs/magvit2_3d_model_config_8x8x8.yaml")
    model = VisionTokenizer(config=cfg, commitment_cost=0, diversity_gamma=0,
                            use_gan=False, use_lecam_ema=False, use_perceptual=False)
    state = torch.load(WEIGHTS_DIR / "3rd_stage.ckpt", map_location="cpu", weights_only=True)
    model.tokenizer.load_state_dict(state, strict=True)
    model.eval()
    model.to("cuda").to(torch.bfloat16)
    model.tokenizer.to(torch.bfloat16)
    return model.tokenizer


@torch.inference_mode()
def encode_to_tokens(magvit_tok, ct: torch.Tensor) -> torch.Tensor:
    """ct: (1, 1, Z, X, Y) on CPU -> quantized tokens (1, 18, T, H, W) on CPU.
    Mirrors run_whole_dataset.py:339-367."""
    from einops import rearrange
    patch = ct.to(torch.bfloat16)
    patch = _center_crop_axis2(patch)                       # trim Z to 1+4n
    h = ct.shape[-2] // 8                                   # spatial_downsample_ratio = 2^3 = 8
    w = ct.shape[-1] // 8
    patch = patch.to("cuda")
    _, encoded_output, _ = magvit_tok.encode(patch, entropy_loss_weight=0.0)
    quantized = magvit_tok.quantize.indices_to_codes(
        indices=encoded_output.indices, project_out=True)
    quantized = rearrange(quantized, "b (t h w) c -> b c t h w", h=h, w=w)
    return quantized.float().cpu()


# ---------- LLaVA + Llama loader (quantized) ----------
def load_llava():
    sys.path.insert(0, "/opt/btb3d/report-generation")
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.train.train import smart_tokenizer_and_embedding_resize
    from llava.constants import (TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER,
                                  TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION)
    from peft import PeftModel

    base_dir = WEIGHTS_DIR / "Llama-3.1-8B-Instruct"
    lora_dir = WEIGHTS_DIR / "checkpoint-43000"

    print(f"[predict] loading Llama-3.1 base from {base_dir} (quant={QUANT})", flush=True)

    kwargs = {
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",        # T4 sm_75 — no flash-attn
        "torch_dtype": torch.bfloat16,
    }
    if QUANT == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype")
    elif QUANT == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype")

    lora_cfg = LlavaConfig.from_pretrained(lora_dir)
    # Match builder.py: temporarily shrink to base vocab so the embed_tokens load lines up.
    lora_cfg.pad_token_id = None
    lora_cfg.vocab_size = lora_cfg.vocab_size - 1 - 4

    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(base_dir, config=lora_cfg, **kwargs)

    # Add pad + 4 task tokens (mirrors builder.py:62-77).
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "<pad>"}, tokenizer=tokenizer, model=model)
    for tok in (TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER,
                TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION):
        tokenizer.add_tokens(tok, special_tokens=True)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Load non_lora_trainables (expanded embed_tokens + mm_projector).
    # mm_projector is not in the base Llama checkpoint, so low_cpu_mem_usage=True
    # leaves its params as meta tensors. Materialize them before load_state_dict
    # so the copy is not a no-op.
    print(f"[predict] loading non_lora_trainables", flush=True)
    nlt = torch.load(lora_dir / "non_lora_trainables.bin", map_location="cpu", weights_only=False)
    nlt = {(k[11:] if k.startswith("base_model.") else k): v for k, v in nlt.items()}
    if any(k.startswith("model.model.") for k in nlt):
        nlt = {(k[6:] if k.startswith("model.") else k): v for k, v in nlt.items()}
    model.model.mm_projector.to_empty(device="cpu")
    model.load_state_dict(nlt, strict=False)
    model.model.mm_projector.to(device="cuda", dtype=torch.bfloat16)

    # Apply LoRA. With bitsandbytes quantization, do NOT merge — merging
    # requires dequantization and busts VRAM. PeftModel.forward dispatches
    # through the LoRA layers correctly without merging.
    print(f"[predict] loading LoRA adapter", flush=True)
    model = PeftModel.from_pretrained(model, str(lora_dir))
    if QUANT == "bf16":
        print(f"[predict] merging LoRA into bf16 base", flush=True)
        model = model.merge_and_unload()
        model = model.to(device="cuda", dtype=torch.bfloat16)

    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_report(tokenizer, model, tokens_npz_arr: torch.Tensor) -> str:
    """tokens_npz_arr: (1, 18, T, H, W) on CPU -> report text."""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates

    # Mirror multigpu_llava_eval.py:116 — transpose to (1, T, H, W, 18).
    image_tensor = tokens_npz_arr.permute(0, 2, 3, 4, 1).to("cuda", dtype=torch.bfloat16)

    conv = conv_templates["llama3"].copy()
    conv.append_message(conv.roles[0], PROMPT)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0).to(model.device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_tensor.shape],
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return text


# ---------- state persistence ----------
def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as e:
            print(f"[predict] could not read checkpoint state ({e}); starting fresh", file=sys.stderr)
    return {"completed": [], "predictions": []}

def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    os.replace(tmp, STATE_PATH)


# ---------- main loop ----------
def write_final_predictions(state: dict) -> None:
    """Write predictions in the grand-challenge wrapped format that
    vlm3d-reportgen-eval:1.0 expects: a top-level JSON array with one
    {"outputs": [{"value": <inner>}]} entry, where <inner> is the canonical
    {"name", "type", "generated_reports": [...]} payload."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inner = {
        "name": "Generated reports",
        "type": "Report generation",
        "version": "1.0",
        "generated_reports": state["predictions"],
    }
    payload = [{"outputs": [{"value": inner}]}]
    PREDICTIONS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[predict] wrote {len(state['predictions'])} entries to {PREDICTIONS_OUT}", flush=True)


def main() -> int:
    for d in (OUTPUT_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[predict] FATAL: input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 1

    inputs = sorted(p for p in INPUT_DIR.rglob("*.nii.gz") if p.is_file())
    if not inputs:
        print(f"[predict] no *.nii.gz under {INPUT_DIR}", file=sys.stderr)
        return 1

    state = load_state()
    completed = set(state["completed"])
    print(f"[predict] found {len(inputs)} volume(s); {len(completed)} already done", flush=True)

    todo = [p for p in inputs if derive_image_name(p) not in completed]
    if not todo:
        write_final_predictions(state)
        return 0

    print(f"[predict] loading models...", flush=True)
    t0 = time.time()
    tokenizer_mag = load_tokenizer()
    tokenizer_llm, model_llm = load_llava()
    print(f"[predict] models loaded in {time.time()-t0:.1f}s", flush=True)

    for i, nii_path in enumerate(todo, 1):
        if _stop_requested:
            print(f"[predict] stopping early (SIGTERM); {len(state['completed'])} completed", flush=True)
            break

        name = derive_image_name(nii_path)
        # Inference runs WITHOUT a try/except wrapper on purpose: if a volume
        # OOMs or otherwise fails, the container must crash loudly so the run
        # fails fast — rather than silently skipping volumes and "succeeding"
        # with partial predictions that waste the whole compute budget.
        t_v = time.time()
        ct = load_and_preprocess(nii_path)
        tokens = encode_to_tokens(tokenizer_mag, ct)
        del ct
        torch.cuda.empty_cache()
        report = generate_report(tokenizer_llm, model_llm, tokens)
        del tokens
        torch.cuda.empty_cache()
        dt = time.time() - t_v
        print(f"[predict] [{i:>4}/{len(todo)}] {name} ({dt:.1f}s, {len(report)} chars)", flush=True)
        state["predictions"].append({"input_image_name": name, "report": report})
        state["completed"].append(name)
        save_state(state)

    write_final_predictions(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
