#!/usr/bin/env python3
"""
process_totalseg_mha_only.py — .mha → .nii.gz → TotalSegmentator → evaluator JSON
=================================================================================

• Scans ONLY .mha files in /input
• Converts each .mha → .nii.gz (temp) for TotalSegmentator
• Runs TS v2 tasks:
    - lung_nodules                  → "Lung nodule"
    - pleural_pericard_effusion     → "Pleural effusion", "Pericardial effusion"
• Extracts 3D axis-aligned bounding boxes (mm); origin rebased to volume min corner
• Writes /output/results.json with ALL labels present:
    - "Lung nodule"             (filled)
    - "Pleural effusion"        (filled)
    - "Pericardial effusion"    (filled)
    - "Consolidation"           []  (empty)
    - "Ground glass opacity"    []  (empty)

Optional env:
  TS_DEVICE=cpu|gpu
  TS_FAST=1
  TS_FORCE_SPLIT=1
  TS_EXTRA="--body_seg"
  TOTALSEG_HOME_DIR=/path/to/.totalsegmentator
  KEEP_TS_OUTPUT=1
"""

from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from skimage import measure
from tqdm import tqdm

# ───────── Paths ───────────────────────────────────────────────────────
VOL_DIR  = Path("/input")
OUT_FILE = Path("/output/results.json")
TMP_ROOT = Path("/output/ts_tmp")
CONV_DIR = TMP_ROOT / "converted"       # where converted .nii.gz go

# ───────── Evaluator labels (always present) ───────────────────────────
EVAL_LABELS = [
    "Lung nodule",
    "Pleural effusion",
    "Pericardial effusion",
    "Consolidation",
    "Ground glass opacity",
]

# TS class → evaluator label mapping per task
TS_TASKS = {
    "lung_nodules": {
        "lung_nodules": "Lung nodule",
    },
    "pleural_pericard_effusion": {
        "pleural_effusion": "Pleural effusion",
        "pericardial_effusion": "Pericardial effusion",
    },
}

# ───────── Utilities ───────────────────────────────────────────────────
def base_stem_mha(p: Path) -> str:
    # strip only .mha (case-insensitive)
    name = p.name
    if name.lower().endswith(".mha"):
        return name[:-4]
    return p.stem

def ensure_dirs() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)

def convert_mha_to_nifti_gz(mha_path: Path) -> Path:
    """
    Convert a .mha to .nii.gz in CONV_DIR and return the new path.
    """
    out_path = CONV_DIR / f"{base_stem_mha(mha_path)}.nii.gz"
    if not out_path.exists():
        img = sitk.ReadImage(str(mha_path))
        sitk.WriteImage(img, str(out_path))
    return out_path

def _autodevice() -> str:
    try:
        import torch  # type: ignore
        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def run_ts(ts_input: Path, out_dir: Path, task: str, device_hint: Optional[str]) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = (device_hint or os.environ.get("TS_DEVICE", "").strip().lower()) or _autodevice()
    cmd = [
        "TotalSegmentator",
        "-i", str(ts_input),
        "-o", str(out_dir),
        "-ta", task,
        "--device", device,
        "--nr_thr_saving", "1",
    ]
    if os.environ.get("TS_FAST") == "1":
        cmd += ["--fast"]
    if os.environ.get("TS_FORCE_SPLIT") == "1":
        cmd += ["--force_split"]
    extra = os.environ.get("TS_EXTRA", "")
    if extra:
        cmd += extra.split()

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[TS] Task '{task}' failed on {ts_input.name}: {e}", file=sys.stderr)
        return False

def ensure_mask_on_ref(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    same = (
            list(mask_img.GetSize()) == list(ref_img.GetSize())
            and np.allclose(mask_img.GetSpacing(),   ref_img.GetSpacing())
            and np.allclose(mask_img.GetOrigin(),    ref_img.GetOrigin())
            and np.allclose(mask_img.GetDirection(), ref_img.GetDirection())
    )
    if same:
        return mask_img
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    return r.Execute(mask_img)

# ───── Physical coord helpers (rebase to volume min corner) ────────────
def _idx_to_phys(x: int, y: int, z: int, itk_img: sitk.Image) -> np.ndarray:
    """Voxel-edge index (x,y,z) → physical coord (mm), honoring spacing+direction+origin."""
    sx, sy, sz = itk_img.GetSpacing()
    origin = np.asarray(itk_img.GetOrigin(), dtype=float)
    direction = np.asarray(itk_img.GetDirection(), dtype=float).reshape(3, 3)
    v = np.array([x * sx, y * sy, z * sz], dtype=float)
    return origin + direction @ v

def _volume_min_corner_phys(itk_img: sitk.Image) -> np.ndarray:
    """Physical coord of the volume's min corner (left–lower–posterior) after direction/origin."""
    size_x, size_y, size_z = itk_img.GetSize()
    corners_idx = [
        (0, 0, 0), (size_x, 0, 0), (0, size_y, 0), (0, 0, size_z),
        (size_x, size_y, 0), (size_x, 0, size_z), (0, size_y, size_z), (size_x, size_y, size_z),
    ]
    corners_phys = np.stack([_idx_to_phys(x, y, z, itk_img) for (x, y, z) in corners_idx], axis=0)
    return corners_phys.min(axis=0)

def boxes_from_mask(mask_zyx: np.ndarray, itk_img: sitk.Image) -> List[List[float]]:
    """
    Connected components → AABB in mm with origin set to the volume's min corner.
    Returns [x_mm, y_mm, z_mm, dx_mm, dy_mm, dz_mm], all starts ≥ 0.
    """
    if mask_zyx.sum() == 0:
        return []

    vol_min = _volume_min_corner_phys(itk_img)

    lab, n_lab = measure.label(mask_zyx.astype(np.uint8), return_num=True, connectivity=1)
    boxes: List[List[float]] = []

    for lab_id in range(1, n_lab + 1):
        zz, yy, xx = np.where(lab == lab_id)  # indices are z,y,x from GetArrayFromImage
        if zz.size == 0:
            continue

        z0, y0, x0 = int(zz.min()), int(yy.min()), int(xx.min())
        z1, y1, x1 = int(zz.max() + 1), int(yy.max() + 1), int(xx.max() + 1)

        # 8 voxel-edge corners of this component, map to patient space
        corners_idx = [
            (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x0, y0, z1),
            (x1, y1, z0), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1),
        ]
        corners_phys = np.stack([_idx_to_phys(x, y, z, itk_img) for (x, y, z) in corners_idx], axis=0)
        pmin = corners_phys.min(axis=0) - vol_min
        pmax = corners_phys.max(axis=0) - vol_min

        dx, dy, dz = (pmax - pmin).tolist()
        x_mm, y_mm, z_mm = pmin.tolist()
        boxes.append([float(x_mm), float(y_mm), float(z_mm), float(dx), float(dy), float(dz)])

    return boxes

def gather_boxes_for_volume(mha_path: Path) -> Dict[str, List[Dict[str, object]]]:
    # Reference geometry: original .mha
    itk_ref = sitk.ReadImage(str(mha_path))
    per_label_boxes: Dict[str, List[List[float]]] = {
        "Lung nodule": [], "Pleural effusion": [], "Pericardial effusion": []
    }

    # Convert .mha → .nii.gz for TS
    ts_input = convert_mha_to_nifti_gz(mha_path)

    # Run TS tasks and extract boxes
    for task, mapping in TS_TASKS.items():
        task_out = TMP_ROOT / f"{base_stem_mha(mha_path)}__{task}"
        ok = run_ts(ts_input, task_out, task, device_hint=None)
        if not ok:
            continue

        for ts_class, eval_label in mapping.items():
            mask_file = task_out / f"{ts_class}.nii.gz"
            if not mask_file.exists():
                alt = task_out / f"{ts_class}.nii"
                mask_file = alt if alt.exists() else mask_file
            if not mask_file.exists():
                print(f"[TS] Missing mask {ts_class} for {mha_path.name}", file=sys.stderr)
                continue

            mask_img = sitk.ReadImage(str(mask_file))
            mask_img = ensure_mask_on_ref(mask_img, itk_ref)
            mask_arr = sitk.GetArrayFromImage(mask_img)  # (z,y,x)
            boxes = boxes_from_mask((mask_arr > 0).astype(np.uint8), itk_ref)
            per_label_boxes[eval_label].extend(boxes)

        if os.environ.get("KEEP_TS_OUTPUT") != "1":
            shutil.rmtree(task_out, ignore_errors=True)

    # Optionally clean converted nifti
    if os.environ.get("KEEP_TS_OUTPUT") != "1":
        try:
            ts_input.unlink(missing_ok=True)
        except Exception:
            pass

    # Wrap into evaluator fields (unused labels empty)
    out: Dict[str, List[Dict[str, object]]] = {}
    for label in EVAL_LABELS:
        if label in per_label_boxes:
            out[label] = [{"bbox_mm": b, "probability": 1.0} for b in per_label_boxes[label]]
        else:
            out[label] = []
    return out

# ───────── Main ────────────────────────────────────────────────────────
def main() -> None:
    ensure_dirs()

    # ONLY .mha (case-insensitive)
    vols = sorted(p for p in VOL_DIR.iterdir() if p.is_file() and p.name.lower().endswith(".mha"))
    if not vols:
        sys.exit("✗ no .mha volumes found in /input")

    all_scans: List[Dict[str, object]] = []
    for vol in tqdm(vols, desc="Volumes (.mha only)"):
        try:
            per_label = gather_boxes_for_volume(vol)
        except Exception as e:
            print(f"[ERROR] Failed on {vol.name}: {e}", file=sys.stderr)
            per_label = {label: [] for label in EVAL_LABELS}

        scan = {"input_image_name": vol.name}
        scan.update(per_label)
        all_scans.append(scan)

    wrapped = [{"outputs": [{"type": "predictions", "value": {"predictions": all_scans}}]}]
    print(wrapped)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(wrapped, indent=2))
    print("✅ results saved →", OUT_FILE)


if __name__ == "__main__":
    main()
