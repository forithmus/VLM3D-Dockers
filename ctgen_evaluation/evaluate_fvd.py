#!/usr/bin/env python3
"""
CT-Net FVD evaluator (single GPU).

• Loads *.mha volumes (generated & GT)
• HU-clips to [-1000,1000]  → scales to [-1,1]
• Centre-pads / crops to (241, 512, 512)
• Runs FVD in chunks of 5 (gen,ref) pairs
• Writes {"FVD_CTNet": <mean_or_null>} to --out_json (optional)
"""
from __future__ import annotations
import argparse, json, pathlib
from typing import List, Optional

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from FVD import fvd_pytorch as fvd                 # pip install fvd_pytorch
import tqdm
from metrics3d import _resize_vol

# ────────────── constants ──────────────
TARGET_DHW = (402, 420, 420)   # final depth, height, width

CHUNK = 4                 # FVD pairs per stratum

# ────────────── helpers ──────────────
def _read_mha(path: pathlib.Path):
    """
    Returns
    -------
    arr : (D, H, W) float32 ndarray
    spacing : (sz, sx, sy) tuple in mm, matching arr’s axis order
    """
    img      = sitk.ReadImage(str(path))
    spacing  = img.GetSpacing()              # (sx, sy, sz) in SimpleITK
    arr      = sitk.GetArrayFromImage(img)   # (sz, sy, sx)
    arr      = arr.astype(np.float32).transpose(0, 2, 1)  # → (sz, sx, sy)

    # re-order spacing to follow (D, H, W) = (z, x, y)
    spacing  = (spacing[2], spacing[0], spacing[1])
    return arr, spacing



def _prep_volume(arr: np.ndarray, spacing: tuple[float, float, float]) -> torch.Tensor:
    """
    HU-clip → resample to TARGET_SPACING → centre pad/crop to TARGET_DHW →
    scale to [-1, 1].
    """
    arr = np.clip(arr, -1000, 1000)
    vol = torch.from_numpy(arr)                  # (D,H,W)

    # 1) spacing-aware resize
    vol = _resize_vol(vol, TARGET_DHW)

    # 3) map from HU in [-1000, 1000] → float in [-1, 1]
    vol = vol / 1000.0
    vol = (vol + 1) / 2
    return vol                                    # (D,H,W) float32 in [0,1]


def load_generated(p: pathlib.Path) -> torch.Tensor:
    arr, sp = _read_mha(p)
    return _prep_volume(arr, sp)


def load_gt(p: pathlib.Path) -> torch.Tensor:
    arr, sp = _read_mha(p)
    return _prep_volume(arr, sp)


def map_to_gt(gen_file: pathlib.Path, gt_root: pathlib.Path) -> pathlib.Path:
    return gt_root / gen_file.name

# ────────────── dataset ──────────────
class VolPairDS(Dataset):
    def __init__(self, gen_dir: pathlib.Path, gt_root: pathlib.Path):
        self.files = sorted(gen_dir.glob("*.mha"))
        self.gt_root = gt_root

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        g = self.files[idx]
        return load_generated(g), load_gt(map_to_gt(g, self.gt_root))

# ────────────── JSON helper ──────────────
def _write_json(path: pathlib.Path, value: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"FVD_CTNet": value}, indent=2))

# ────────────── main evaluation ──────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VolPairDS(args.generated_dir, args.gt_root)

    if len(ds) == 0:
        print(f"⚠️  No *.mha files found in {args.generated_dir}")
        if args.out_json:
            _write_json(args.out_json, None)
        return

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)

    gb, rb, scores = [], [], []
    for gen, ref in tqdm.tqdm(loader, desc="Volumes"):
        gen, ref = gen.contiguous(), ref.contiguous()

        print(gen.shape, flush=True)
        gb.append(gen[0,0].cpu())
        rb.append(ref[0,0].cpu())

        if len(gb) == CHUNK:
            g_np = torch.stack(gb).numpy().astype(np.float32)
            r_np = torch.stack(rb).numpy().astype(np.float32)
            print(g_np.shape, flush=True)
            scores.append(
                fvd.compute_fvd(r_np, g_np, model="ctnet", device="cuda"))
            gb = []
            rb = []
            #torch.cuda.empty_cache()
    if gb:
        g_np = torch.stack(gb).cpu().numpy()
        r_np = torch.stack(rb).cpu().numpy()
        scores.append(float(
            fvd.compute_fvd(r_np, g_np, model="ctnet", device="cuda")))



    mean_fvd = float(np.mean(scores)) if scores else None
    print(f"\n✓  Mean FVD (CT-Net) over {len(scores)} strata: {mean_fvd}")

    if args.out_json:
        _write_json(args.out_json, mean_fvd)
        print(f"✓  JSON saved to {args.out_json}")

# ────────────── CLI ──────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_dir", type=pathlib.Path, required=True,
                    help="Directory with generated *.mha volumes")
    ap.add_argument("--gt_root", type=pathlib.Path, required=True,
                    help="Directory with ground-truth *.mha volumes")
    ap.add_argument("--workers", type=int, default=0,
                    help="Data-loading workers (0 = inline)")
    ap.add_argument("--out_json", type=pathlib.Path,
                    help="Write {'FVD_CTNet': mean_score} JSON here")
    main(ap.parse_args())