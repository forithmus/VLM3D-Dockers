#!/usr/bin/env python3
"""
ctchat_pipeline.py (no‑CLI version)
===================================

*No runtime flags required.*
The script assumes a fixed container layout:

```
/input   ← directory with CT volumes (*.mha / *.nii / *.nii.gz)
/output  ← writable directory for the resulting JSON

/opt/app/models/clip_visual_encoder.pth   ← CTViT weights
/opt/app/models/llava-llama3_1_8B_ctclip-finetune_256-lora_2gpus                  ← CT‑CHAT checkpoint dir
```
"""
from __future__ import annotations

import json, random, sys, warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image
import pandas as pd

# ---------------------------------------------------------------------------
#  HARD‑WIRED PATHS & PARAMETERS — EDIT HERE IF NEEDED
# ---------------------------------------------------------------------------
VOL_DIR        = Path("/input")
OUT_FILE       = Path("/output/results.json")
SAVE_NPZ       = False                  # set True to cache embeddings
DEVICE         = "cuda"
TEMPERATURE    = 0.0
MAX_NEW_TOKENS = 512

CTVIT_WEIGHTS  = Path("/opt/app/models/clip_visual_encoder.pth")
CTCHAT_PATH    = Path("/opt/app/models/llava-llama3_1_8B_ctclip-finetune_256-lora_2gpus/checkpoint-114000")
LLAMA_PATH = Path("/opt/app/llama")
CTCHAT_BASE    = None                   # set to a base model if needed

# ---------------------------------------------------------------------------
#  PROMPT BANK
# ---------------------------------------------------------------------------
QUESTIONS = [
    "Can you generate the report for the following chest CT image?",
    "Please provide the radiology report for the chest CT image mentioned.",
    "I need the radiology report for the given chest CT image.",
    "Could you create a report for this chest CT scan?",
    "Would you mind generating the radiology report for the specified chest CT image?",
    "Please generate the report for the chest CT image provided.",
    "Can you produce the radiology report for the attached chest CT image?",
    "I need a detailed report for the given chest CT image.",
    "Could you write the radiology report for this chest CT scan?",
    "Please give the radiology report for the specified chest CT image.",
    "Generate radiology report for the CT.",
    "Produce the report for this CT image.",
    "Write a radiology report for the following CT scan.",
    "Create a report for this chest CT.",
    "Provide the radiology report for this CT image.",
    "Can you generate the report for the following chest CT volume?",
    "Please provide the radiology report for the chest CT volume mentioned.",
    "I need the radiology report for the given chest CT volume.",
    "Could you create a report for this chest CT volume?",
    "Would you mind generating the radiology report for the specified chest CT volume?",
    "Please generate the report for the chest CT volume provided.",
    "Can you produce the radiology report for the attached chest CT volume?",
    "I need a detailed report for the given chest CT volume.",
    "Could you write the radiology report for this chest CT volume?",
    "Please give the radiology report for the specified chest CT volume.",
    "Generate radiology report for the CT volume.",
    "Produce the report for this CT volume.",
    "Write a radiology report for the following CT volume.",
    "Create a report for this chest CT volume.",
    "Provide the radiology report for this CT volume.",
    "Can you generate the report for the following chest CT scan?",
    "Please provide the radiology report for the chest CT scan mentioned.",
    "I need the radiology report for the given chest CT scan.",
    "Could you create a report for this chest CT scan?",
    "Would you mind generating the radiology report for the specified chest CT scan?",
    "Please generate the report for the chest CT scan provided.",
    "Can you produce the radiology report for the attached chest CT scan?",
    "I need a detailed report for the given chest CT scan.",
    "Could you write the radiology report for this chest CT scan?",
    "Please give the radiology report for the specified chest CT scan.",
    "Generate radiology report for the CT scan.",
    "Produce the report for this CT scan.",
    "Write a radiology report for the following CT scan.",
    "Create a report for this chest CT scan.",
    "Provide the radiology report for this CT scan."
]

# ---------------------------------------------------------------------------
#  IMPORT LIBRARIES (CTViT + CT‑CHAT)
# ---------------------------------------------------------------------------
try:
    from transformer_maskgit import CTViT
except ImportError as e:
    print("✗ transformer_maskgit is not installed.")
    sys.exit(1)

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import TextStreamer

# ---------------------------------------------------------------------------
#  PRE‑PROCESSING CONSTANTS
# ---------------------------------------------------------------------------
HU_MIN, HU_MAX   = -1000, 1000
TARGET_SHAPE     = (480, 480, 240)  # (H, W, D)
TARGET_SPACING   = (1.5, 0.75, 0.75)  # (z, x, y) mm

# --------------------------- Helper Functions ----------------------------- #

def resize_vol(vol: torch.Tensor, spacing: tuple[float, float, float]):
    z, x, y = spacing
    current_shape = vol.shape[2:]
    current_spacing = spacing
    print(current_spacing)
    scaling_factors = [
        current_spacing[i] / TARGET_SPACING[i] for i in range(len(current_spacing))
    ]
    new_shape = [
        int(current_shape[i] * scaling_factors[i]) for i in range(len(current_shape))
    ]
    print(new_shape)
    return F.interpolate(vol, size=new_shape, mode="trilinear", align_corners=False)



# ----------------------------- Model Loading ------------------------------ #

def load_ctvit(path: Path) -> "CTViT":
    if not path.exists():
        raise FileNotFoundError(f"CTViT weights not found at {path}")
    model = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    ).to(DEVICE).eval()
    model.load(str(path))
    return model


def load_ctchat(path: Path, base: Path | None) -> tuple:
    if not path.exists():
        raise FileNotFoundError(f"CT‑CHAT checkpoint not found at {path}")
    disable_torch_init()
    name = get_model_name_from_path(str(path))
    tok, model, _, _ = load_pretrained_model(str(path), str(base), name, device=DEVICE)
    if "llama-2" in name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llama3"
    conv_mode == "llama3"
    return tok, model, conv_mode


# ----------------------------- Pipeline Steps ----------------------------- #

def encode_volume(path: Path, encoder) -> torch.Tensor:
    itk = sitk.ReadImage(str(path))
    name = str(path)
    img = sitk.GetArrayFromImage(itk).astype("float32")
    print(img.max())
    print(img.min())

    vol = np.clip(img, HU_MIN, HU_MAX)
    t   = torch.from_numpy(vol).cpu().unsqueeze(0).unsqueeze(0)

    sx, sy, sz = itk.GetSpacing()
    print(sx,sy,sz)
    t   = resize_vol(t, (sz, sx, sy))
    img_data = t[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (img_data) / 1000.0
    slices = []

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    tensor = tensor.permute(2, 1, 0)

    tensor = tensor.unsqueeze(0).unsqueeze(0)


    
    with torch.inference_mode():
        return encoder(tensor.cuda(), return_encoded_tokens=True).half()


def make_prompt() -> str:
    return f"<image>\n{random.choice(QUESTIONS)}<report_generation>"


def generate_report(tok, model, conv_mode, emb: torch.Tensor) -> str:
    text = make_prompt()
    print(text)
    print("\n")
    conv = conv_templates[conv_mode].copy()


    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)
    ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        out = model.generate(ids, images=emb, image_sizes=[emb.numel()], do_sample=TEMPERATURE > 0, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer)
    return tok.decode(out[0], skip_special_tokens=True).strip()


# ----------------------------- Main --------------------------------------- #

def main():
    encoder = load_ctvit(CTVIT_WEIGHTS)
    tok, chat, conv_mode = load_ctchat(CTCHAT_PATH, LLAMA_PATH)

    vols = sorted([p for p in VOL_DIR.iterdir() if p.suffix.lower() in {".mha", ".nii", ".nii.gz"}])
    if not vols:
        print("✗ No CT volumes found in", VOL_DIR)
        sys.exit(1)

    reports: List[Dict[str, str]] = []
    for v in tqdm(vols, desc="Volumes", unit="scan"):
        emb = encode_volume(v, encoder)

        rep = generate_report(tok, chat, conv_mode, emb)
        reports.append({"input_image_name": v.name.split(".")[0], "report": rep})
        tqdm.write("✓ " + v.name)

    #OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Assemble JSON output
    result = {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": reports,
        "version": {"major": 1, "minor": 0},
    }

    with OUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)
    print("Saved →", OUT_FILE)


if __name__ == "__main__":
    main()
