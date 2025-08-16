#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abnormality-Localisation Evaluation (strict, robust)

- 0 / 1 / N boxes allowed per pathology.
- Each box must have exactly 6 numbers; malformed -> ValueError.
- If a “box” is actually a list-of-boxes (len!=6 but inner lists are 6),
  we unwrap correctly instead of failing.
- When there are GT boxes but no predictions/matches, assign worst scores
  (IoU=0.0, Distance=volume diagonal) instead of None.
"""

import ast
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from tqdm import tqdm as _tqdm
    tqdm = lambda x, **k: _tqdm(x, **k)
except ImportError:
    tqdm = lambda x, **k: x

# ───────── Paths ───────── #
INPUT_DIR   = Path("/input")
OUTPUT_DIR  = Path("/output")
CODE_DIR    = Path("/opt/app")
GT_CSV      = CODE_DIR / "ground-truth" / "bbox_ground_truth.csv"
FINAL_JSON  = OUTPUT_DIR / "metrics.json"

# ───────── Constants ───────── #
LABELS = [
    "Pericardial effusion",
    "Pleural effusion",
    "Consolidation",
    "Ground glass opacity",
    "Lung nodule",
]
IOU_THRESH       = 0.10
FROC_BUDGETS_FP  = [0.5, 1, 2, 4]
WORST_IOU        = 0.0

# Volume size (mm) and derived worst distance (diagonal)
VOL_DIMS_MM      = np.array([512, 512, 256], dtype=float)
WORST_DISTANCE   = float(np.linalg.norm(VOL_DIMS_MM))  # 768.0 mm

GT_FIELD_MAP = {
    "pericardial_effusion": "Pericardial effusion",
    "pleural_effusion":     "Pleural effusion",
    "consolidation":        "Consolidation",
    "ggd":                  "Ground glass opacity",
    "nodule":               "Lung nodule",
}

# ───────── Helpers ───────── #
def _first_json(dir_: Path) -> Path:
    files = sorted(dir_.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json found in {dir_}")
    return files[0]

def _parse_str(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return json.loads(s)

def _ensure_box_list(obj):
    """
    Return list[list[6 floats]] or raise.

    Handles:
      - single flat list -> wraps
      - list of lists
      - strings / numpy arrays
      - accidental extra nesting (e.g. whole list-of-boxes treated as one box)
    """
    if isinstance(obj, str):
        obj = _parse_str(obj)

    # single flat list -> wrap
    if isinstance(obj, (list, tuple)) and all(isinstance(x, (int, float)) for x in obj):
        obj = [obj]

    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"Expected list/tuple, got {type(obj)}: {obj}")

    fixed = []

    def _fix_one(box):
        if isinstance(box, str):
            box = _parse_str(box)
        if hasattr(box, "tolist"):
            box = box.tolist()

        # flat box
        if isinstance(box, (list, tuple)) and all(isinstance(x, (int, float)) for x in box):
            if len(box) != 6:
                raise ValueError(f"Malformed bbox (len={len(box)}): {box}")
            return [[float(v) for v in box]]

        # list of boxes
        if isinstance(box, (list, tuple)) and len(box) > 0 and isinstance(box[0], (list, tuple)):
            out = []
            for b in box:
                if len(b) != 6 or not all(isinstance(v, (int, float)) for v in b):
                    raise ValueError(f"Malformed bbox inside nested list: {b}")
                out.append([float(v) for v in b])
            return out

        raise ValueError(f"Bbox not list-like or wrong shape: {box}")

    for elem in obj:
        fixed.extend(_fix_one(elem))

    return fixed  # list of [6]

def _boxes_from_json_fixed(pred_json: Path) -> pd.DataFrame:
    with pred_json.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data[0]["outputs"][0]["value"]["predictions"]:
        acc = Path(item["input_image_name"]).stem.replace(".mha", "")
        missing = [lab for lab in LABELS if lab not in item]
        if missing:
            raise ValueError(f"Missing keys for scan {acc}: {', '.join(missing)}")

        for label in LABELS:
            for box in item[label]:
                if "bbox_mm" not in box:
                    raise ValueError(f"Missing 'bbox_mm' in {acc}, {label}: {box}")
                bxs = _ensure_box_list(box["bbox_mm"])  # returns list of [6]
                prob = float(box.get("probability", 1.0))
                for b in bxs:
                    rows.append({
                        "AccessionNo": acc,
                        "label":       label,
                        "x":           b[0],
                        "y":           b[1],
                        "z":           b[2],
                        "dx":          b[3],
                        "dy":          b[4],
                        "dz":          b[5],
                        "probability": prob,
                    })

    return pd.DataFrame(rows)

def _load_ground_truth(gt_csv: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(gt_csv)
    required = {"id", *GT_FIELD_MAP.keys()}
    if not required.issubset(df_raw.columns):
        raise ValueError(f"Ground-truth CSV missing columns: {required - set(df_raw.columns)}")

    records = []
    for _, row in df_raw.iterrows():
        acc = str(row["id"])
        for col, label in GT_FIELD_MAP.items():
            cell = row[col]
            if pd.isna(cell) or cell == "" or cell == "[]":
                continue
            for b in _ensure_box_list(cell):
                records.append({
                    "AccessionNo": acc,
                    "label":       label,
                    "x_mm":        b[0],
                    "y_mm":        b[1],
                    "z_mm":        b[2],
                    "dx_mm":       b[3],
                    "dy_mm":       b[4],
                    "dz_mm":       b[5],
                    "probability": 1.0,
                })
    return pd.DataFrame(records)

def _iou_3d(a, b) -> float:
    ax, ay, az, adx, ady, adz = a
    bx, by, bz, bdx, bdy, bdz = b
    ax2, ay2, az2 = ax + adx, ay + ady, az + adz
    bx2, by2, bz2 = bx + bdx, by + bdy, bz + bdz
    inter_dx = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_dy = max(0.0, min(ay2, by2) - max(ay, by))
    inter_dz = max(0.0, min(az2, bz2) - max(az, bz))
    inter_vol = inter_dx * inter_dy * inter_dz
    if inter_vol == 0:
        return 0.0
    vol_a = adx * ady * adz
    vol_b = bdx * bdy * bdz
    return inter_vol / (vol_a + vol_b - inter_vol)

def _centroid(box):
    x, y, z, dx, dy, dz = box
    return np.array([x + dx/2, y + dy/2, z + dz/2])

def _match_one_case(pred_boxes: np.ndarray, gt_boxes: np.ndarray):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return [], 0, 0
    if len(pred_boxes) == 0:
        return [], 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return [], len(pred_boxes), 0

    cost = np.ones((len(pred_boxes), len(gt_boxes)))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            cost[i, j] = 1.0 - _iou_3d(p, g)

    rows, cols = linear_sum_assignment(cost)

    tp_pairs, matched_pred, matched_gt = [], set(), set()
    for r, c in zip(rows, cols):
        iou = 1.0 - cost[r, c]
        if iou >= IOU_THRESH:
            dist = float(np.linalg.norm(_centroid(pred_boxes[r]) - _centroid(gt_boxes[c])))
            tp_pairs.append((iou, dist))
            matched_pred.add(r)
            matched_gt.add(c)

    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes)  - len(matched_gt)
    return tp_pairs, fp, fn

def evaluate(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
    pred_grp = pred_df.groupby(["AccessionNo", "label"])
    gt_grp   = gt_df.groupby(["AccessionNo", "label"])

    iou_vals, dist_vals = defaultdict(list), defaultdict(list)
    fp_per_scan, gt_per_scan = defaultdict(float), defaultdict(int)

    scans = set(pred_df["AccessionNo"]).union(gt_df["AccessionNo"])
    for scan in tqdm(scans, desc="Evaluating"):
        for label in LABELS:
            p = pred_grp.get_group((scan, label))[["x","y","z","dx","dy","dz"]].to_numpy() if (scan, label) in pred_grp.groups else np.empty((0, 6))
            g = gt_grp.get_group((scan, label))[["x_mm","y_mm","z_mm","dx_mm","dy_mm","dz_mm"]].to_numpy() if (scan, label) in gt_grp.groups else np.empty((0, 6))
            pairs, fp, fn = _match_one_case(p, g)
            for iou, dist in pairs:
                iou_vals[label].append(iou)
                dist_vals[label].append(dist)
            fp_per_scan[scan] += fp
            gt_per_scan[scan] += len(g)

    # Count GT per label to know which labels are present in this dataset
    gt_counts_by_label = gt_df["label"].value_counts().to_dict()

    metrics = {"iou": {"per_pathology": {}, "macro": None},
               "distance": {"per_pathology": {}, "macro": None},
               "froc": {}}

    # Per-pathology with worst scores when GT exists but no matches
    for label in LABELS:
        has_gt = gt_counts_by_label.get(label, 0) > 0

        if iou_vals[label]:
            i_mean = float(np.mean(iou_vals[label]))
        else:
            i_mean = WORST_IOU if has_gt else None

        if dist_vals[label]:
            d_mean = float(np.mean(dist_vals[label]))
        else:
            d_mean = WORST_DISTANCE if has_gt else None

        metrics["iou"]["per_pathology"][label]      = i_mean
        metrics["distance"]["per_pathology"][label] = d_mean

    # Macros are over labels that actually have GT (i.e., not None)
    iou_macro_vals  = [v for v in metrics["iou"]["per_pathology"].values() if v is not None]
    dist_macro_vals = [v for v in metrics["distance"]["per_pathology"].values() if v is not None]

    metrics["iou"]["macro"]      = float(np.mean(iou_macro_vals))  if iou_macro_vals  else None
    metrics["distance"]["macro"] = float(np.mean(dist_macro_vals)) if dist_macro_vals else None

    # FROC
    total_gt = gt_df.shape[0]
    preds_sorted = pred_df.sort_values("probability", ascending=False).reset_index(drop=True)

    for budget in FROC_BUDGETS_FP:
        tp_found, fp_counter = set(), defaultdict(float)
        for _, row in preds_sorted.iterrows():
            scan = row["AccessionNo"]
            if fp_counter[scan] >= budget:
                continue

            cand = gt_df[(gt_df["AccessionNo"] == scan) & (gt_df["label"] == row["label"])]
            best_iou, best_gt_idx = 0.0, None
            for g_idx, g_row in cand.iterrows():
                if g_idx in tp_found:
                    continue
                iou = _iou_3d(
                    row[["x","y","z","dx","dy","dz"]].to_numpy(),
                    g_row[["x_mm","y_mm","z_mm","dx_mm","dy_mm","dz_mm"]].to_numpy()
                )
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, g_idx

            if best_iou >= IOU_THRESH:
                tp_found.add(best_gt_idx)
            else:
                fp_counter[scan] += 1

        sens = len(tp_found) / total_gt if total_gt else 0.0
        if budget == 0.5:
            budget = "05"
        metrics["froc"][f"{str(budget)}_fp_per_image"] = sens

    return metrics

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pred_json = _first_json(INPUT_DIR)
    pred_df   = _boxes_from_json_fixed(pred_json)
    gt_df     = _load_ground_truth(GT_CSV)
    results   = evaluate(pred_df, gt_df)
    FINAL_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("✅ metrics.json written to", FINAL_JSON, file=sys.stderr)

if __name__ == "__main__":
    main()
