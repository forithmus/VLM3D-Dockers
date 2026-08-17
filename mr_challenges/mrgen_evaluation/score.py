"""
MR Volume Generation — Evaluation Orchestrator
=================================================

Platform contract:

    /input/
      predictions/    <- participant-generated .nii.gz volumes
      ground_truth/   <- reference MR volumes (flat directory)
    /output/
      metrics.json    <- {"metrics": {...}}

No prompts.json is needed — the modality is parsed directly from the ground
truth filename ("{study_uid}_{modality}-raw-{plane}.nii.gz"); your generated
volume must use the same filename as its target.

MEMORY/SCALE NOTE: at ~14k volumes, collecting everything in memory before
computing FID would exhaust RAM, so volumes are processed ONE PAIR AT A TIME:
each (real, pred) pair is read, basic metrics (MSE/PSNR/SSIM) and FID
features are extracted immediately, and the volume memory is released (see
fid_2p5d.FIDAccumulator). Only one volume pair plus the small accumulated
feature vectors are ever resident.

NETWORK NOTE: execution runs offline, so the squeezenet1_1 weights used for
FID features are downloaded at BUILD time and baked into the image.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from modality_filter import is_scored_modality
from fid_2p5d import FIDAccumulator
from metrics_basic import compute_basic_metrics

INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/input"))
PREDICTIONS_DIR = Path(os.environ.get("PREDICTIONS_DIR", str(INPUT_ROOT / "predictions")))
GROUND_TRUTH_DIR = Path(os.environ.get("GROUND_TRUTH_DIR", str(INPUT_ROOT / "ground_truth")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))

WORST_SCORE = {"MSE": 1.0, "PSNR": 0.0, "SSIM": -1.0}

# FID feature çıkarımını her N vakada bir stderr'e ilerleme logu bas —
# 13930 dosyalık bir koşuda "container donmuş mu" belirsizliğini önlemek için.
PROGRESS_EVERY = int(os.environ.get("EVAL_PROGRESS_EVERY", "25"))
# How many volume pairs to decode ahead of the scoring loop.
PREFETCH = int(os.environ.get("EVAL_PREFETCH", "4"))


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj, dtype=np.float32)


def list_ground_truth_files() -> list[Path]:
    return sorted(GROUND_TRUTH_DIR.glob("*.nii.gz")) + sorted(GROUND_TRUTH_DIR.glob("*.nii"))


def build_prediction_index() -> dict[str, Path]:
    """Index the predictions directory once (single listing) instead of probing
    the FUSE mount per ground-truth file."""
    if not PREDICTIONS_DIR.exists():
        return {}
    return {p.name: p for p in PREDICTIONS_DIR.iterdir() if p.is_file()}


def find_prediction_file(gt_filename: str, index: dict[str, Path]) -> Path | None:
    hit = index.get(gt_filename)
    if hit is not None:
        return hit
    stem = gt_filename.replace(".nii.gz", "").replace(".nii", "")
    for ext in (".nii.gz", ".nii"):
        hit = index.get(f"{stem}{ext}")
        if hit is not None:
            return hit
    return None


def main() -> int:
    if not GROUND_TRUTH_DIR.exists():
        print(f"HATA: ground truth klasörü bulunamadı: {GROUND_TRUTH_DIR}", file=sys.stderr)
        return 1
    if not PREDICTIONS_DIR.exists():
        print(f"UYARI: predictions klasörü yok: {PREDICTIONS_DIR} (hepsi missing sayılacak)", file=sys.stderr)

    all_gt_files = list_ground_truth_files()
    scored_files = [p for p in all_gt_files if is_scored_modality(p.name)]

    n_total = len(all_gt_files)
    n_scored = len(scored_files)
    n_excluded = n_total - n_scored
    print(
        f"Ground truth: {n_total} toplam dosya, {n_scored} skorlanacak "
        f"(T1w/T2w/FLAIR/SWI), {n_excluded} kapsam dışı (MRA/DWI/ADC vb.).",
        flush=True,
    )

    fid_acc = FIDAccumulator(device="auto")

    pred_index = build_prediction_index()
    print(f"Predictions: {len(pred_index)} dosya indekslendi.", flush=True)

    per_case = []
    missing_count = 0

    matched = []
    for gt_path in scored_files:
        pred_path = find_prediction_file(gt_path.name, pred_index)
        if pred_path is None:
            missing_count += 1
            per_case.append({"file": gt_path.name, "missing_output": True, **WORST_SCORE})
        else:
            matched.append((gt_path, pred_path))
    print(f"Eşleşen çift: {len(matched)}, çıktısı olmayan: {missing_count}", flush=True)

    def _read_pair(item):
        gt_path, pred_path = item
        return gt_path, load_volume(gt_path), load_volume(pred_path)

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        pending = deque()
        it = iter(matched)
        for _ in range(PREFETCH):
            nxt = next(it, None)
            if nxt is None:
                break
            pending.append(pool.submit(_read_pair, nxt))

        done_n = 0
        while pending:
            fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(_read_pair, nxt))

            done_n += 1
            if done_n % PROGRESS_EVERY == 0 or done_n == len(matched):
                print(f"İlerleme: {done_n}/{len(matched)}", flush=True)

            try:
                gt_path, real_vol, fake_vol = fut.result()
            except Exception as e:
                print(f"UYARI: okunamadı ({e}), missing olarak işaretlendi.", file=sys.stderr)
                missing_count += 1
                continue

            basic = compute_basic_metrics(real_vol, fake_vol)
            per_case.append({"file": gt_path.name, "missing_output": False, **basic})

            # FID feature'ları hemen çıkar, hacimleri elden bırak (bellek).
            fid_acc.add_pair(real_vol, fake_vol)
            del real_vol, fake_vol

    fid_scores = fid_acc.finalize()

    def _agg(key: str) -> float:
        vals = [c[key] for c in per_case if not c["missing_output"]]
        return float(np.mean(vals)) if vals else float("nan")

    metrics = {
        "MSE_mean": _agg("MSE"),
        "PSNR_mean": _agg("PSNR"),
        "SSIM_mean": _agg("SSIM"),
        **fid_scores,
        "n_total_files": n_total,
        "n_scored_files": n_scored,
        "n_excluded_out_of_scope_modality": n_excluded,
        "n_missing_outputs": missing_count,
        "dice": _agg("SSIM"),  # NOT: gercek dice degil, platform primary-metric uyumlulugu icin SSIM_mean'in kopyasi
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_case": per_case}, f, indent=2, ensure_ascii=False)

    print("\n=== Özet (metrics) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nmetrics.json yazıldı: {OUTPUT_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
