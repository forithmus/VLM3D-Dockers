#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_fid_2p5d_ct.py

Compute 2.5D Frechet Inception Distance (FID) for CT volumes using 3 orthogonal planes.
This version uses a custom Backbone class, loads the model from a local checkpoint,
and includes a manual loader for .mha files.

**It now correctly uses array-based MONAI transforms for manually loaded data.**
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import monai
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from torchvision.models import resnet50
from monai.metrics.fid import FIDMetric
# --- Import ARRAY-BASED transforms ---
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    CenterSpatialCrop,
    SpatialPad,
)
from tqdm import tqdm

# Configure logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants
PLANES: tuple[Literal["xy", "yz", "xz"], ...] = ("xy", "yz", "xz")
SUPPORTED_EXTENSIONS = ["*.nii", "*.nii.gz", "*.mha"]

# Custom Loader
def load_volume_as_tensor(path: Path) -> torch.Tensor:
    try:
        img_itk = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img_itk).astype(np.float32)
        return torch.from_numpy(arr)
    except Exception as e:
        logger.error(f"Failed to load image at {path} using SimpleITK.")
        raise e

# Feature Extractor
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base_model.children())[:9])
    def forward(self, x):
        features = self.backbone(x)
        return features.flatten(start_dim=1)

# Slice processing
def normalize_slice_batch(x: torch.Tensor) -> torch.Tensor:
    min_vals = x.amin(dim=(2, 3), keepdim=True)
    max_vals = x.amax(dim=(2, 3), keepdim=True)
    x = (x - min_vals) / (max_vals - min_vals + 1e-10)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    x = x - imagenet_mean
    return x

@torch.no_grad()
def get_features_for_volume(vol: torch.Tensor, model: nn.Module, plane: Literal["xy", "yz", "xz"], center_slices_ratio: float, batch_size: int, device: torch.device) -> torch.Tensor:
    vol = vol.to(device)
    vol = vol.repeat(1, 3, 1, 1, 1)
    vol = vol[:, [2, 1, 0], ...]
    if plane == "xy": slices = vol.unbind(dim=2)
    elif plane == "yz": slices = vol.unbind(dim=4)
    else: slices = vol.unbind(dim=3)
    if 0.0 < center_slices_ratio < 1.0:
        n = len(slices)
        start, end = int((1.0 - center_slices_ratio) / 2.0 * n), int((1.0 + center_slices_ratio) / 2.0 * n)
        slices = slices[start:end]
    all_features = []
    for i in range(0, len(slices), batch_size):
        batch = torch.cat(slices[i : i + batch_size], dim=0)
        batch = normalize_slice_batch(batch)
        features = model(batch)
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0) if all_features else torch.empty(0, 2048)

# File Discovery
def find_image_files(directory: Path) -> list[Path]:
    files = []
    for ext in SUPPORTED_EXTENSIONS: files.extend(directory.glob(ext))
    if not files: logger.warning(f"No supported image files found in {directory}")
    return sorted(files)

# Main Execution
def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Model
    model = Backbone().to(device)
    logger.info(f"Loading weights from local checkpoint: {args.model_ckpt}")
    if not args.model_ckpt.exists():
        logger.error(f"Model checkpoint not found at {args.model_ckpt}!")
        sys.exit(1)
    ckpt = torch.load(args.model_ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    if all(key.startswith('backbone.') for key in state_dict.keys()):
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    model.backbone.load_state_dict(state_dict, strict=True)
    model.eval()

    # 2. Build ARRAY-BASED Preprocessing Pipeline
    t_shape = [int(x) for x in args.target_shape.split("x")]

    post_load_transforms = Compose([
        SpatialPad(spatial_size=t_shape, method="end", mode="constant", constant_values=-1000) if args.enable_padding else nn.Identity(),
        CenterSpatialCrop(roi_size=t_shape) if args.enable_center_cropping else nn.Identity(),
        ScaleIntensityRange(a_min=-1000.0, a_max=1000.0, b_min=-1.0, b_max=1.0, clip=True)
    ])

    # 3. Find and Process Files
    real_files = find_image_files(Path(args.real_dir))
    synth_files = find_image_files(Path(args.synth_dir))
    synth_map = {f.name: f for f in synth_files}
    file_pairs = [(p, synth_map[p.name]) for p in real_files if p.name in synth_map]

    if not file_pairs:
        logger.error("No matching file pairs found."); sys.exit(1)
    if args.num_images > 0: file_pairs = file_pairs[:args.num_images]
    logger.info(f"Found {len(file_pairs)} matching image pairs to process.")

    # 4. Extract Features
    real_features, synth_features = {p: [] for p in PLANES}, {p: [] for p in PLANES}
    for real_path, synth_path in tqdm(file_pairs, desc="Extracting Features", unit="vol"):
        # --- Apply array-based transforms directly to the tensor ---
        real_tensor = load_volume_as_tensor(real_path)
        real_vol = post_load_transforms(real_tensor).unsqueeze(0)

        synth_tensor = load_volume_as_tensor(synth_path)
        synth_vol = post_load_transforms(synth_tensor).unsqueeze(0)
        # -----------------------------------------------------------

        for plane in PLANES:
            real_features[plane].append(get_features_for_volume(real_vol, model, plane, args.center_slices_ratio, args.batch_size, device))
            synth_features[plane].append(get_features_for_volume(synth_vol, model, plane, args.center_slices_ratio, args.batch_size, device))

    # 5. Calculate FID
    results, fid_metric = {}, FIDMetric()
    logger.info("Calculating FID scores...")
    for plane in PLANES:
        all_real = torch.cat(real_features[plane], dim=0)
        all_synth = torch.cat(synth_features[plane], dim=0)
        logger.info(f"Plane {plane.upper()}: Real slices={all_real.shape[0]}, Synth slices={all_synth.shape[0]}")
        fid_value = fid_metric(all_synth, all_real).item()
        results[f"FID_{plane.upper()}"] = fid_value
        print(f"  FID_{plane.upper()}: {fid_value:.4f}")

    mean_fid = sum(results.values()) / len(results)
    results["FID_2p5D_mean"] = mean_fid
    print(f"FID_2p5D_mean: {mean_fid:.4f}")

    if args.out_json:
        try:
            args.out_json.parent.mkdir(exist_ok=True, parents=True)
            args.out_json.write_text(json.dumps(results, indent=2))
            logger.info(f"Results saved to {args.out_json}")
        except Exception as e:
            logger.error(f"Could not write JSON output: {e}")

def parse_args() -> argparse.Namespace:
    # CLI parsing is unchanged
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real_dir", type=Path, required=True, help="Directory with real/ground-truth 3D volumes (NIfTI or MHA).")
    p.add_argument("--synth_dir", type=Path, required=True, help="Directory with synthetic/generated 3D volumes (NIfTI or MHA).")
    p.add_argument("--out_json", type=Path, help="Optional path to save the results as a JSON file.")
    g_pre = p.add_argument_group("Preprocessing Options")
    g_pre.add_argument("--target_shape", type=str, default="512x512x512", help="Target shape for padding/cropping, as 'DxHxW'.")
    g_pre.add_argument("--enable_padding", action="store_true", help="Pad volumes to target_shape.")
    g_pre.add_argument("--enable_center_cropping", action="store_true", help="Center-crop volumes to target_shape.")
    g_fid = p.add_argument_group("FID Options")
    g_fid.add_argument("--model_ckpt", type=Path, default="/opt/app/ResNet50.pt", help="Path to the local ResNet50 model checkpoint (.pt file).")
    g_fid.add_argument("--center_slices_ratio", type=float, default=0.4, help="Ratio of center slices to use from each axis.")
    g_fid.add_argument("--num_images", type=int, default=-1, help="Maximum number of images to process. -1 for all.")
    g_fid.add_argument("--batch_size", type=int, default=32, help="Batch size for processing 2D slices on the GPU.")
    g_fid.add_argument("--device", type=str, default="cuda", help="PyTorch device to use.")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())