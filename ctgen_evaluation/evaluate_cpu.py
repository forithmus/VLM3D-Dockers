#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_cpu.py
===============

CPU-only image-quality metrics (PSNR • SSIM • MSE) on *.mha* CT volumes.

• Both generated and ground-truth volumes are                   ┐
      ▸ resampled to TARGET_SPACING (trilinear)                │
      ▸ centre-cropped / sym-padded to TARGET_DHW              │
      ▸ mapped to [0,1] intensity range                        ┘

No external CSV is required; voxel spacing is read from each MetaImage header.
"""
from __future__ import annotations

import argparse, json, pathlib
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.regression import MeanSquaredError
from tqdm import tqdm

# ──────────────────────── constants ────────────────────────
HU_MIN, HU_MAX   = -1000.0, 1000.0               # clamp for GT volumes
TARGET_SPACING   = (1.5, 0.75, 0.75)             # (tz, ty, tx) in mm
TARGET_DHW       = (241, 512, 512)               # (D, H, W) after crop/pad

# ──────────────────────── helpers ──────────────────────────
def _read_mha(p: pathlib.Path) -> np.ndarray:
    """
    Read *.mha* as (D, H, W) float32 ndarray. SimpleITK returns (z,y,x),
    so we transpose to (z,y,x) → (D,H,W) = (z,x,y) with .transpose(0,2,1).
    """
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    return arr.transpose(0, 2, 1)


def _resize_vol(vol: torch.Tensor,
                curr_spacing: Tuple[float, float, float],
                tgt_spacing:  Tuple[float, float, float]) -> torch.Tensor:
    """
    Trilinear resample a (D,H,W) tensor from curr_spacing to tgt_spacing.

    curr_spacing / tgt_spacing order corresponds to vol axes: (z, y, x).
    """
    scale_fac = [cs / ts for cs, ts in zip(curr_spacing, tgt_spacing)]
    new_shape = [max(1, int(round(s * f))) for s, f in zip(vol.shape, scale_fac)]

    # vol[None,None] → (1,1,D,H,W) for F.interpolate, then squeeze back
    return F.interpolate(vol[None, None],
                         size=new_shape,
                         mode="trilinear",
                         align_corners=False)[0, 0]


def _center_crop_or_pad(v: torch.Tensor) -> torch.Tensor:
    """Centre-crop, then symmetric-pad to TARGET_DHW = (D,H,W)."""
    D_t, H_t, W_t = TARGET_DHW
    d, h, w = v.shape

    # ---------- centre crop ----------
    if d > D_t: v = v[(d - D_t) // 2 : (d - D_t) // 2 + D_t]
    if h > H_t: v = v[:, (h - H_t) // 2 : (h - H_t) // 2 + H_t, :]
    if w > W_t: v = v[:, :, (w - W_t) // 2 : (w - W_t) // 2 + W_t]

    # ---------- symmetric pad (pad order: W1,W2,H1,H2,D1,D2) ----------
    pad = [0, W_t - v.shape[2], 0, H_t - v.shape[1], 0, D_t - v.shape[0]]
    return F.pad(v, pad, value=HU_MIN)


# ──────────────────────── loaders ─────────────────────────
def _load_gen(p: pathlib.Path) -> torch.Tensor:
    """
    Generated volume → (D,H,W) float32 in [0,1].

    Assumes intensities are approximately in [-1,1] **before** mapping.
    """
    itk = sitk.ReadImage(str(p))
    sx, sy, sz = itk.GetSpacing()                  # (x,y,z) in mm

    arr = _read_mha(p)
    arr = median_filter(arr, size=3)               # simple denoise
    v = torch.from_numpy(arr).permute(2, 0, 1)     # (D,H,W)

    v = _resize_vol(v, (sz, sy, sx), TARGET_SPACING)
    v = _center_crop_or_pad(v)

    v = (torch.clamp(v, -1.0, 1.0) + 1.0) / 2.0    # → [0,1]
    return v                                       # (D,H,W)


def _load_gt(p: pathlib.Path) -> torch.Tensor:
    """Ground-truth volume → (D,H,W) float32 in [0,1]."""
    itk = sitk.ReadImage(str(p))
    sx, sy, sz = itk.GetSpacing()

    v = torch.from_numpy(_read_mha(p)).permute(2, 0, 1)  # (D,H,W)
    v = _resize_vol(v, (sz, sy, sx), TARGET_SPACING)
    v = _center_crop_or_pad(v)

    v = (torch.clamp(v, HU_MIN, HU_MAX) / HU_MAX + 1.0) / 2.0
    return v


# ──────────────────────── dataset ─────────────────────────
class PairDS(Dataset):
    """Pairs *.mha* in `gen_dir` with identically-named files in `gt_root`."""
    def __init__(self, gen_dir: pathlib.Path, gt_root: pathlib.Path):
        self.gen_files = sorted(gen_dir.glob("*.mha"))
        if not self.gen_files:
            raise RuntimeError(f"No *.mha files found in {gen_dir}")
        self.gt_root = gt_root

    def __len__(self) -> int:
        return len(self.gen_files)

    def __getitem__(self, idx):
        g_path = self.gen_files[idx]
        return _load_gen(g_path), _load_gt(self.gt_root / g_path.name)


# ──────────────────────── main ────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_dir", type=pathlib.Path, required=True,
                    help="Directory with generated *.mha* volumes")
    ap.add_argument("--gt_root",       type=pathlib.Path, required=True,
                    help="Directory with ground-truth *.mha* volumes")
    ap.add_argument("--threads",       type=int, default=8,
                    help="torch.set_num_threads")
    ap.add_argument("--workers",       type=int, default=4,
                    help="torch DataLoader num_workers")
    ap.add_argument("--out_json",      type=pathlib.Path,
                    help="Write scores to this JSON as well as stdout")
    args = ap.parse_args()


    loader = DataLoader(
        PairDS(args.generated_dir, args.gt_root),
        batch_size=1,
        num_workers=args.workers,
        pin_memory=False,
    )

    psnr = PeakSignalNoiseRatio().cpu()
    ssim = StructuralSimilarityIndexMeasure().cpu()
    mse  = MeanSquaredError().cpu()

    psnr_list, ssim_list, mse_list = [], [], []

    for g, r in tqdm(loader, desc="Volumes"):
        # g, r shapes: (1, D, H, W) after collate → squeeze batch dim
        g = g[0].unsqueeze(1).repeat(1, 3, 1, 1)  # (D,3,H,W)
        r = r[0].unsqueeze(1).repeat(1, 3, 1, 1)  # (D,3,H,W)

        psnr_list.append(psnr(g, r).item())
        ssim_list.append(ssim(g, r).item())
        mse_list.append(mse(g, r).item())
        print(psnr_list)

    res = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "MSE":  float(np.mean(mse_list)),
    }

    print(json.dumps(res, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
