#!/usr/bin/env python3
"""
Master evaluation for the MR-RATE report-generation track.

Mirrors the CT reportgen evaluation container's contract, with one structural
difference: clinical labels are extracted from the generated text by a local
Gemma model via vLLM (the same pipeline that produced the ground-truth labels)
instead of a fine-tuned BERT classifier. Extraction covers all 74 NeuroVFM
diagnoses; scoring is restricted to the 32-label challenge subset.

/input/
  predictions/    first *.json = participant submission
                  {"generated_reports": [{"input_image_name", "report"}]}
  ground_truth/   ground_truth.json (reference reports)
                  ground_truth.csv  (32-label binary)
                  _assets/gemma/    (extraction model, staged with the GT)
/output/
  metrics.json    {"generation": ..., "classification": ..., "crg": ...}
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

INPUT_DIR = Path("/input/predictions")
GT_DIR = Path("/input/ground_truth")
OUTPUT_DIR = Path("/output")
CODE = Path("/opt/app")

GT_JSON = GT_DIR / "ground_truth.json"
GT_CSV = GT_DIR / "ground_truth.csv"
# On the platform, /input/ground_truth is a GCS FUSE mount. vLLM mmaps the
# 59 GiB Gemma weights, and mmap over FUSE is unusably slow — so the model
# is staged to local SSD before extraction (stage_model(), ~5 min once).
MODEL_SRC = GT_DIR / "_assets" / "gemma"
MODEL = Path("/tmp/gemma")


def stage_model() -> Path:
    """Copy the extraction model from the FUSE-mounted GT assets to local disk.
    Skipped when already staged (retry) or when running outside the platform
    with a local GT directory (copytree is still correct, just cheap)."""
    if MODEL.exists():
        return MODEL
    t0 = time.time()
    print(f"[eval] staging {MODEL_SRC} -> {MODEL} (local disk for vLLM mmap)", flush=True)
    shutil.copytree(MODEL_SRC, MODEL)
    print(f"[eval] model staged in {time.time() - t0:.0f}s", flush=True)
    return MODEL

PRED_CSV = OUTPUT_DIR / "extracted_labels.csv"
CLS_JSON = OUTPUT_DIR / "classification_scores.json"
CRG_JSON = OUTPUT_DIR / "crg_scores.json"
NLG_JSON = OUTPUT_DIR / "nlg_scores.json"
FINAL = OUTPUT_DIR / "metrics.json"


def run(script, *args):
    cmd = [sys.executable, str(CODE / script), *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = sorted(INPUT_DIR.glob("*.json"))
    if not preds:
        raise FileNotFoundError(f"No *.json submission in {INPUT_DIR}/")
    pred_json = preds[0]
    print("submission:", pred_json, flush=True)

    # 1. NLG first: cheap, and still produces scores if extraction dies later.
    run("nlg_metrics.py", "--pred_json", pred_json, "--gt_json", GT_JSON,
        "--out_json", NLG_JSON)

    # 2. LLM label extraction from generated text
    stage_model()
    run("extract_labels_vllm.py",
        "--pred_json", pred_json,
        "--diagnoses_json", CODE / "neurovfm_mri_diagnoses.json",
        "--keep_labels", CODE / "keep32.txt",
        "--model_path", MODEL,
        "--out_csv", PRED_CSV)

    # 3. classification + CRG on the 32-label subset
    run("calc_scores.py", "--pred_csv", PRED_CSV, "--gt_csv", GT_CSV,
        "--out_json", CLS_JSON)
    run("crg_score.py", "--pred_csv", PRED_CSV, "--gt_csv", GT_CSV,
        "--out_json", CRG_JSON)

    def load(p):
        return json.loads(Path(p).read_text())

    combined = {
        "generation": load(NLG_JSON),
        "classification": load(CLS_JSON),
        "crg": load(CRG_JSON),
    }
    FINAL.write_text(json.dumps(combined, indent=2))
    print("all metrics ->", FINAL, flush=True)


if __name__ == "__main__":
    main()
