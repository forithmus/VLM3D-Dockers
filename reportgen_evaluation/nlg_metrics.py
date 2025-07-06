#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute NLG scores between prediction and ground-truth JSON files.

Called from evaluation.py like:
    python nlg_metrics.py --pred_json /input/inferred.json \
                          --gt_json   /opt/app/ground-truth/ground_truth.json \
                          --out_json  /output/nlg_scores.json
"""

import argparse
import json
from pathlib import Path

import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# ------------------------------------------------------------------ #
# Optional METEOR → may fail if Java is missing; handled gracefully
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
def compute_scores(gts, res):
    scorers = [
        (Bleu(4),   ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(),   "ROUGE_L"),
        (Cider(),   "CIDEr"),
        (Meteor(),  "METEOR")
    ]

    out = {}
    for scorer, names in scorers:
        try:
            score, _ = scorer.compute_score(gts, res, verbose=0)
        except FileNotFoundError:            # METEOR → java not found
            print("⚠️  Java not found → skipping METEOR")
            continue
        except TypeError:
            score, _ = scorer.compute_score(gts, res)

        if isinstance(names, list):          # BLEU returns 4 scores
            for s, n in zip(score, names):
                out[n] = s
            # --- NEW: mean BLEU ------------------------------------------------
            out["BLEU_mean"] = sum(score) / len(score)
            # ------------------------------------------------------------------
        else:
            out[names] = score
    return out


# ------------------------------------------------------------------ #
def run(pred_json: Path, gt_json: Path, out_json: Path):
    with open(pred_json, "r", encoding="utf-8") as f:
        in_json = json.load(f)
        raw = in_json[0]["outputs"][0]["value"]
        pred_items = raw["generated_reports"]
    with open(gt_json, "r", encoding="utf-8") as f:
        gt_items = json.load(f)["generated_reports"]

    pred_map = {x["input_image_name"].rsplit(".", 1)[0]: x["report"]
                for x in pred_items}
    gt_map   = {x["input_image_name"].rsplit(".", 1)[0]: x["report"]
                for x in gt_items}

    gts, recs = {}, {}
    for idx, key in enumerate(tqdm.tqdm(sorted(set(gt_map) & set(pred_map)))):
        gts[idx]  = [gt_map[key].replace("\n", "").replace(" . ", ".").replace(" .", ".")]
        recs[idx] = [pred_map[key].replace("\n", "")
                     .replace("<|eot_id|>", "")
                     .replace("\"", "")
                     .replace("_", "")
                     .replace(" . ", ".")
                     .replace(" .", ".")]

    scores = compute_scores(gts, recs)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print("NLG metrics →", out_json)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True,
                    help="Path to inferred.json (predictions)")
    ap.add_argument("--gt_json", required=True,
                    help="Path to ground_truth.json")
    ap.add_argument("--out_json", required=True,
                    help="Where to write the metrics")
    args = ap.parse_args()

    run(Path(args.pred_json), Path(args.gt_json), Path(args.out_json))
