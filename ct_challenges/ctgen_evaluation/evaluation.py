#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py — Forithmus-port of the VLM3D ct-volume-generation eval.

Forithmus mounts the participant's generated volumes at /input/predictions
and the host ground truth at /input/ground_truth. Everything else (CT-CLIP,
FVD, ResNet50, BiomedVLP-CXR-BERT, data_input.xlsx) is baked into the image.

/input/
  predictions/   - participant submission: *.nii.gz (.mha also accepted)
  ground_truth/  - host ground truth: *.mha/*.nii.gz volumes (real CT, hidden)
/output/
  metrics.json   - merged scores (FVD, CLIP, FID 2.5D, optional CPU)
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

try:
    import SimpleITK as sitk
except ImportError:
    print("Warning: SimpleITK not found. FID pre-processing will fail.", file=sys.stderr)
    sitk = None

try:
    import torch
except ImportError:
    print("Warning: PyTorch not found. Cannot determine GPU count for torchrun.", file=sys.stderr)
    torch = None

# ───────────────────────── paths ─────────────────────────
INPUT_DIR         = Path("/input/predictions")    # participant .nii.gz here
GT_ROOT_MHA       = Path("/input/ground_truth")   # host volumes here
OUTPUT_DIR        = Path("/output")
CODE_DIR          = Path("/opt/app")
TEMP_PRED_DIR     = Path("/tmp/unzipped_mha")

# FID 2.5D temp
FID_TEMP_DIR       = OUTPUT_DIR / "fid_temp"
GT_ROOT_NIFTI      = FID_TEMP_DIR / "real_nifti"
INPUT_DIR_NIFTI    = FID_TEMP_DIR / "synth_nifti"
FID_FEATURE_DIR    = FID_TEMP_DIR / "features"
GT_FILELIST_TXT    = FID_TEMP_DIR / "real_files.txt"
INPUT_FILELIST_TXT = FID_TEMP_DIR / "synth_files.txt"

# Scripts + prompts xlsx (still baked in)
PROMPT_XLSX = CODE_DIR / "data_input.xlsx"
FVD_SCRIPT  = CODE_DIR / "evaluate_fvd.py"
CLIP_SCRIPT = CODE_DIR / "evaluate_clip.py"
CPU_SCRIPT  = CODE_DIR / "evaluate_cpu.py"
FID_SCRIPT  = CODE_DIR / "compute_fid_2-5d_ct.py"

# JSON outputs
FVD_JSON   = OUTPUT_DIR / "fvd_scores.json"
CLIP_JSON  = OUTPUT_DIR / "clip_scores.json"
CPU_JSON   = OUTPUT_DIR / "quality_scores.json"
FID_JSON   = OUTPUT_DIR / "fid_scores.json"
FINAL_JSON = OUTPUT_DIR / "metrics.json"

FLOAT = r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
REGEX = {
    "FVD":            re.compile(r"FVD.*?:\s*" + FLOAT),
    "CLIPScore":      re.compile(r"CLIPScore\s*:\s*" + FLOAT),
    "CLIPScore_I2I":  re.compile(r"CLIPScore_I2I\s*:\s*" + FLOAT),
    "CLIPScore_mean": re.compile(r"CLIPScore_mean\s*:\s*" + FLOAT),
    "PSNR":           re.compile(r"PSNR.*?:\s*" + FLOAT),
    "SSIM":           re.compile(r"SSIM.*?:\s*" + FLOAT),
    "MSE":            re.compile(r"MSE.*?:\s*" + FLOAT),
    "FID_2p5D_Avg":   re.compile(r"FID Avg:\s*" + FLOAT),
    "FID_2p5D_XY":    re.compile(r"FID XY:\s*" + FLOAT),
    "FID_2p5D_XZ":    re.compile(r"FID ZX:\s*" + FLOAT),
    "FID_2p5D_YZ":    re.compile(r"FID YZ:\s*" + FLOAT),
}


def _run(cmd: List[str], out_json: Optional[Path]) -> str:
    if out_json:
        cmd += ["--out_json", str(out_json)]
    print(">>", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
            lines.append(line)
    finally:
        proc.stdout.close()
        rc = proc.wait()
    out = "".join(lines)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=out)
    return out


def _extract(key: str, txt: str) -> Optional[float]:
    m = REGEX.get(key)
    return float(m.search(txt).group(1)) if m and m.search(txt) else None


def _safe_write_json(path: Path, payload: dict) -> None:
    clean = {k: v for k, v in payload.items() if v is not None}
    if clean:
        path.write_text(json.dumps(clean, indent=2))
    else:
        print(f"No data for {path.name}, skipping write.", file=sys.stderr)


_VOL_EXTS = (".nii.gz", ".nii", ".mha")


def _case_stem(p: Path) -> str:
    n = p.name
    for ext in _VOL_EXTS:
        if n.lower().endswith(ext):
            return n[: -len(ext)]
    return p.stem


def _list_volumes(d: Path):
    return sorted(p for p in d.rglob("*")
                  if p.is_file() and p.name.lower().endswith(_VOL_EXTS))


def _convert_and_list_files(src_dir: Path, nifti_dir: Path, filelist: Path) -> None:
    """Normalize a directory of volumes (.nii.gz or .mha) to .nii.gz under
    nifti_dir, keyed by case stem. SimpleITK reads both formats."""
    if not sitk:
        raise ImportError("SimpleITK is required for volume→NIfTI conversion.")
    nifti_dir.mkdir(parents=True, exist_ok=True)
    files = _list_volumes(src_dir)
    print(f"Preparing {len(files)} volumes from {src_dir} → NIfTI...", flush=True)
    names = []
    for f in files:
        tgt = nifti_dir / (_case_stem(f) + ".nii.gz")
        if not tgt.exists():
            sitk.WriteImage(sitk.ReadImage(str(f)), str(tgt))
        names.append(tgt.name)
    filelist.write_text("\n".join(names))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Locate generated volumes. The algorithm now writes loose .nii.gz to
    # /output (one per prompt). .mha and a .zip of volumes are still accepted
    # for backward compatibility with older submissions.
    all_vols = _list_volumes(INPUT_DIR)
    if all_vols:
        generated_dir = INPUT_DIR
        print(f"Found {len(all_vols)} volume(s) in {INPUT_DIR}", flush=True)
    else:
        zipfiles = sorted(INPUT_DIR.rglob("*.zip"))
        if zipfiles:
            zip_path = zipfiles[0]
            print(f"Extracting {zip_path.name} → {TEMP_PRED_DIR}", flush=True)
            TEMP_PRED_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.lower().endswith(_VOL_EXTS):
                        out = TEMP_PRED_DIR / Path(member).name
                        out.write_bytes(zf.read(member))
            generated_dir = TEMP_PRED_DIR
        else:
            generated_dir = INPUT_DIR

    if not GT_ROOT_MHA.exists() or not _list_volumes(GT_ROOT_MHA):
        raise FileNotFoundError(
            f"No volume files found under {GT_ROOT_MHA}. Was ground-truth mounted by the platform?"
        )

    FID_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1) FVD
    try:
        out = _run([sys.executable, str(FVD_SCRIPT),
                    "--generated_dir", str(generated_dir),
                    "--gt_root",       str(GT_ROOT_MHA)], FVD_JSON)
    except Exception as e:
        print(f"FVD failed: {e}", file=sys.stderr)
        out = str(e)
    if not FVD_JSON.exists():
        _safe_write_json(FVD_JSON, {"FVD_CTNet": _extract("FVD", out)})

    # 2) CLIP
    try:
        out = _run([sys.executable, str(CLIP_SCRIPT),
                    "--generated_dir", str(generated_dir),
                    "--gt_root",       str(GT_ROOT_MHA),
                    "--prompt_xlsx",   str(PROMPT_XLSX)], CLIP_JSON)
    except Exception as e:
        print(f"CLIP failed: {e}", file=sys.stderr)
        out = str(e)
    if not CLIP_JSON.exists():
        _safe_write_json(CLIP_JSON, {
            "CLIPScore":      _extract("CLIPScore", out),
            "CLIPScore_I2I":  _extract("CLIPScore_I2I", out),
            "CLIPScore_mean": _extract("CLIPScore_mean", out),
        })

    # 3) FID 2.5D
    try:
        print("\n--- Prepare 2.5D FID data ---", flush=True)
        _convert_and_list_files(GT_ROOT_MHA, GT_ROOT_NIFTI, GT_FILELIST_TXT)
        _convert_and_list_files(generated_dir, INPUT_DIR_NIFTI, INPUT_FILELIST_TXT)
        n_gpus = torch.cuda.device_count() if torch and torch.cuda.is_available() else 1
        out = _run([
            "torchrun", f"--nproc_per_node={n_gpus}", str(FID_SCRIPT),
            "--real_dataset_root", str(GT_ROOT_NIFTI),
            "--real_filelist",     str(GT_FILELIST_TXT),
            "--real_features_dir", "real_features",
            "--synth_dataset_root", str(INPUT_DIR_NIFTI),
            "--synth_filelist",     str(INPUT_FILELIST_TXT),
            "--synth_features_dir", "synth_features",
            "--output_root",        str(FID_FEATURE_DIR),
            "--target_shape",       "512x512x512",
            "--enable_padding",     "True",
            "--enable_center_cropping", "True",
            "--enable_resampling_spacing", "1.0x1.0x1.0",
            "--num_images", "100",
            "--model_name", "squeezenet1_1",
        ], None)
    except Exception as e:
        print(f"FID 2.5D failed: {e}", file=sys.stderr)
        out = str(e)
    _safe_write_json(FID_JSON, {
        "FID_2p5D_Avg": _extract("FID_2p5D_Avg", out),
        "FID_2p5D_XY":  _extract("FID_2p5D_XY",  out),
        "FID_2p5D_XZ":  _extract("FID_2p5D_XZ",  out),
        "FID_2p5D_YZ":  _extract("FID_2p5D_YZ",  out),
    })

    # 4) Merge results
    merged = {}
    for p in (FVD_JSON, CLIP_JSON, FID_JSON, CPU_JSON):
        if p.exists():
            try:
                merged.update(json.loads(p.read_text()))
            except Exception as e:
                print(f"Could not parse {p.name}: {e}", file=sys.stderr)

    if not merged:
        merged = {"status": "failed", "error": "No metrics were calculated."}

    # The phase's score_json_path is $.metrics.FVD_CTNet, so wrap accordingly.
    payload = {"metrics": merged}
    FINAL_JSON.write_text(json.dumps(payload, indent=2))
    print("\nAll metrics written to", FINAL_JSON)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
