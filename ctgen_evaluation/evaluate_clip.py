#!/usr/bin/env python3
"""
CT-CLIP evaluation (image–text and image–image) on a single GPU.

* expects identical *.mha file names in generated_dir/ and gt_root/
* NO external meta-data CSV ― slope / intercept already baked in
* Spacing is fetched from each image header via SimpleITK

Outputs
-------
CLIPScore          – image ⟷ text
CLIPScore_I2I      – generated image ⟷ reference image
"""
from __future__ import annotations
import argparse, json, pathlib, warnings
from typing import Tuple

import SimpleITK as sitk            # pip install SimpleITK
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import BertTokenizer, BertModel

from transformer_maskgit import CTViT
from ct_clip            import CTCLIP
import tqdm


import pathlib, typing

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ───────────────────────── constants ──────────────────────────
TARGET_SPACING: typing.Tuple[float, float, float] = (1.5, 0.75, 0.75)   # (z, x, y)
TARGET_DHW:     typing.Tuple[int,   int,   int]   = (240, 480, 480)     # (D, H, W)
HU_MIN, HU_MAX                                = -1000.0, 1000.0
PAD_VALUE                                     = -1.0                   # keep outside HU range


# ───────────────────────── helpers ────────────────────────────
def _read_mha(p: pathlib.Path) -> np.ndarray:
    """MetaImage → float32 ndarray (D, H, W)."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)  # (Z, Y, X)
    return arr.transpose(0, 2, 1)                                            # (D, H, W)

def _resize_array(vol: torch.Tensor,
                  cur_spacing: typing.Tuple[float, float, float]) -> torch.Tensor:
    """Resample `vol` (D,H,W) to TARGET_SPACING with trilinear interpolation."""
    d, h, w  = vol.shape
    scale    = [cur_spacing[i] / TARGET_SPACING[i] for i in range(3)]
    new_size = [int(round(dim * sc)) for dim, sc in zip((d, h, w), scale)]
    return F.interpolate(
        vol.unsqueeze(0).unsqueeze(0),      # → (1,1,D,H,W)
        size=new_size,
        mode="trilinear",
        align_corners=False
    )[0, 0]                                 # back to (D,H,W)

def _centre_crop_or_pad(v: torch.Tensor) -> torch.Tensor:
    """Centre-crop or pad to TARGET_DHW with constant value."""
    D, H, W = TARGET_DHW
    pad = [
        (W - v.shape[2]) // 2,  W - v.shape[2] - (W - v.shape[2]) // 2,
        (H - v.shape[1]) // 2,  H - v.shape[1] - (H - v.shape[1]) // 2,
        (D - v.shape[0]) // 2,  D - v.shape[0] - (D - v.shape[0]) // 2,
        ]
    v = F.pad(v, pad, value=PAD_VALUE)      # PyTorch expects (W1, W2, H1, H2, D1, D2)
    return v[:D, :H, :W]                    # safe-guard against tiny rounding mismatches

def _load_vol(p: pathlib.Path) -> torch.Tensor:
    """Read, resample, crop/pad and clip a single *.mha* volume."""
    arr = _read_mha(p)                                    # (D,H,W)
    arr = np.clip(arr, HU_MIN, HU_MAX)

    itk_img   = sitk.ReadImage(str(p))                    # need spacing (x,y,z!)
    sx, sy, sz = itk_img.GetSpacing()                     # (x, y, z)
    v = torch.from_numpy(arr).permute(0, 1, 2)            # already (D,H,W)

    v = _resize_array(v, (sz, sy, sx))                    # spacing order → (z,x,y)
    v = _centre_crop_or_pad(v)

    # Optional: scale to [0,1] if your downstream model needs that
    # v = (v - HU_MIN) / (HU_MAX - HU_MIN)

    return v

def _clip_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return max(
        100*pairwise_cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0,0].item(),
        0.0,
        )

class VolTextDS(Dataset):
    def __init__(self,
                 gen_dir: pathlib.Path,
                 gt_root: pathlib.Path,
                 prompt_xlsx: pathlib.Path):
        import pandas as pd
        self.gen_files = sorted(gen_dir.glob("*.mha"))
        if not self.gen_files:
            raise RuntimeError(f"No *.mha in {gen_dir}")
        self.gt_root  = gt_root
        self.prompts  = pd.read_excel(prompt_xlsx, engine="openpyxl")

    def __len__(self): return len(self.gen_files)

    def __getitem__(self, idx):
        g = self.gen_files[idx]
        row = self.prompts[self.prompts["Names"] == g.name]
        prompt = "" if row.empty else row["Text_prompts"].iloc[0]
        return _load_vol(g), prompt, _load_vol(self.gt_root/g.name)


# ───────────────────────── main ───────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated_dir", type=pathlib.Path, required=True)
    p.add_argument("--gt_root",       type=pathlib.Path, required=True)
    p.add_argument("--prompt_xlsx",   type=pathlib.Path, default="data_input.xlsx")
    p.add_argument("--workers",       type=int, default=0)
    p.add_argument("--out_json",      type=pathlib.Path)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build CT-CLIP
    tok = BertTokenizer.from_pretrained("/opt/app/models/BiomedVLP-CXR-BERT-specialized")
    txt = BertModel.from_pretrained("/opt/app/models/BiomedVLP-CXR-BERT-specialized").to(dev)
    img = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8,
    ).to(dev)
    clip = CTCLIP(
        image_encoder=img, text_encoder=txt,
        dim_image=294_912, dim_text=768, dim_latent=512,
    ).to(dev)
    clip.load("/opt/app/models/CT-CLIP_v2.pt")
    clip.eval()

    ds  = VolTextDS(args.generated_dir, args.gt_root, args.prompt_xlsx)
    ld  = DataLoader(ds, batch_size=1, num_workers=args.workers)

    s_txt, s_i2i = [], []
    for gv, prompt, rv in tqdm.tqdm(ld):
        gv, rv = gv.to(dev), rv.to(dev)
        gv_r = gv / 1000.0
        rv_r = rv / 1000.0
        print(gv_r.max(), gv_r.min())
        #gv_r   = _resize_vol(gv, (240,480,480))*2.-1.
        #rv_r   = _resize_vol(rv, (240,480,480))*2.-1.
        with torch.no_grad():
            toks = tok(prompt[0], return_tensors="pt", padding="max_length",
                       truncation=True, max_length=512).to(dev)
            txt_lat, img_lat_g, _ = clip(toks, gv_r.unsqueeze(0),
                                         return_latents=True, device=dev)
            _, img_lat_r, _       = clip(toks, rv_r.unsqueeze(0),
                                         return_latents=True, device=dev)
        s_txt.append(_clip_score(img_lat_g[0], txt_lat[0]))
        s_i2i.append(_clip_score(img_lat_g[0], img_lat_r[0]))

    res = {"CLIPScore": float(np.mean(s_txt)), "CLIPScore_I2I": float(np.mean(s_i2i)), "CLIPScore_mean": ((float(np.mean(s_txt)) + float(np.mean(s_i2i))) / 2)}
    print(json.dumps(res, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    import numpy as np
    main()
