#!/usr/bin/env python3
"""
ctclip_classifier_pipeline.py
=============================

Abnormality‑classification inference script for the competition docker.
The pipeline **shares exactly the same CT pre‑processing procedure** as
`ctchat_pipeline.py` so that both tasks consume identically prepared
volumes.

Fixed container layout
----------------------
```
/input                                ← CT volumes (*.mha / *.nii / *.nii.gz)
/output                               ← will receive results.json

/opt/app/models/clip_visual_encoder.pth            ← CTViT weights
/opt/app/models/BiomedVLP-CXR-BERT-specialized     ← Biomed‑BERT tokenizer + ckpt
/opt/app/models/ctclip_classifier.pth              ← ImageLatentsClassifier weights
```

Output schema (must match challenge spec)
----------------------------------------
```json
{
  "name": "Generated probabilities",
  "type": "Abnormality Classification",
  "version": {"major": 1, "minor": 0},
  "predictions": [
    {
      "input_image_name": "<volume>.nii.gz",
      "scores": {"Medical material": 0.00, "Arterial wall calcification": 0.42, …}
    }
  ]
}
```

The script runs on **CUDA** when available and falls back to CPU.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

from PIL import Image          # <-- put with your other imports

SLICE_DIR = Path("/output/mid_slices")   # <-- define near OUT_FILE
SLICE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  HARD‑WIRED PATHS & CONSTANTS
# ---------------------------------------------------------------------------
VOL_DIR          = Path("/input")
OUT_FILE         = Path("/output/results.json")

TEXT_MODEL_PATH  = Path("/opt/app/models/BiomedVLP-CXR-BERT-specialized")
CLASSIFIER_PATH  = Path("/opt/app/models/CT_LiPro_v2.pt")

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

HU_MIN, HU_MAX   = -1000, 1000
TARGET_SHAPE     = (480, 480, 240)            # (H, W, D)
TARGET_SPACING   = (1.5, 0.75, 0.75)           # (z, x, y) mm

PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
NUM_CLASSES = len(PATHOLOGIES)

# ---------------------------------------------------------------------------
#  THIRD‑PARTY IMPORTS (fail fast if missing)
# ---------------------------------------------------------------------------
try:
    from transformers import BertTokenizer, BertModel
    from transformer_maskgit import CTViT
    from ct_clip import CTCLIP
except Exception as e:  # pragma: no cover
    raise SystemExit(f"✗ Required library missing: {e}")

# ---------------------------------------------------------------------------
#  MODEL DEFINITIONS
# ---------------------------------------------------------------------------

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))


class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)  # Assuming trained_model.image_latents_dim gives the size of the image_latents

    def forward(self, latents=False, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents, _ = self.trained_model(*args, **kwargs)
        image_latents = self.relu(image_latents)
        if latents:
            return image_latents
        image_latents = self.dropout(image_latents)  # Apply dropout on the latents

        return self.classifier(image_latents)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict, strict=True)

# ---------------------------------------------------------------------------
#  PRE‑PROCESSING (exactly mirrors ctchat_pipeline.py)
# ---------------------------------------------------------------------------

def resize_vol(vol: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
    """Trilinear resampling to TARGET_SPACING (ctchat identical)."""
    z, x, y = spacing
    scaling = [spacing[i] / TARGET_SPACING[i] for i in range(3)]
    D, H, W = vol.shape[2:]
    new_shape = [int(D * scaling[0]), int(H * scaling[1]), int(W * scaling[2])]
    return F.interpolate(vol, size=new_shape, mode="trilinear", align_corners=False)


def preprocess_volume(path: Path) -> torch.Tensor:
    """Applies the same clipping, resampling, normalisation, crop/pad as ctchat."""
    itk  = sitk.ReadImage(str(path))
    img  = sitk.GetArrayFromImage(itk).astype("float32")  # z, y, x
    print(img.max())
    print(img.min())
    vol  = np.clip(img, HU_MIN, HU_MAX)
    print(vol.shape)
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    sx, sy, sz = itk.GetSpacing()                        # spacing (x,y,z)
    t = resize_vol(t, (sz, sx, sy))                      # need (z,x,y)
    print(t.shape)
    tensor = t[0, 0].permute(1, 2, 0)                    # (H,W,D)
    tensor = tensor / 1000.0                             # HU → [−1,1]

    slices = []

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


    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())
    return tensor

# ---------------------------------------------------------------------------
#  MODEL LOADING UTILITIES
# ---------------------------------------------------------------------------

def build_clip_stack() -> ImageLatentsClassifier:
    """Constructs CT‑CLIP + classifier with pretrained weights."""
    tokenizer    = BertTokenizer.from_pretrained(TEXT_MODEL_PATH, local_files_only=True, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(TEXT_MODEL_PATH, local_files_only=True)
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    ).eval()

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294_912,  # 512 × 576 patches
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    ).eval()
    classifier = ImageLatentsClassifier(clip, latent_dim=512, num_classes=NUM_CLASSES).eval()
    classifier.load(CLASSIFIER_PATH)
    classifier.to(DEVICE)
    classifier.eval()
    classifier.tokenizer = tokenizer  # handy reference
    return classifier

# ---------------------------------------------------------------------------
#  INFERENCE
# ---------------------------------------------------------------------------

def predict(classifier: ImageLatentsClassifier, vol_tensor: torch.Tensor) -> Dict[str, float]:
    text_tokens = classifier.tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(DEVICE)
    logits      = classifier(False,text_tokens, vol_tensor.to(DEVICE), device = DEVICE)
    probs       = sigmoid(logits).detach().cpu().numpy()[0]
    print(probs)
    return {p: float(prob) for p, prob in zip(PATHOLOGIES, probs)}

# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    vols = sorted([p for p in VOL_DIR.iterdir() if p.suffix.lower() in {".mha", ".nii", ".nii.gz"}])
    if not vols:
        raise SystemExit(f"✗ No CT volumes found in {VOL_DIR}")

    classifier = build_clip_stack()

    predictions: List[Dict[str, str | Dict[str, float]]] = []
    for v in tqdm(vols, desc="Volumes", unit="scan"):
        vol_tensor = preprocess_volume(v)
        scores     = predict(classifier, vol_tensor)
        predictions.append({"input_image_name": v.name.replace(".mha","").replace(".nii",""), "probabilities": scores})
        tqdm.write("✓ " + v.name)

    result = {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"major": 1, "minor": 0},
        "predictions": predictions,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)
    print("Saved →", OUT_FILE)


if __name__ == "__main__":
    main()
