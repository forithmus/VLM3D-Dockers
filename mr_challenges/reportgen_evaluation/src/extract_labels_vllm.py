#!/usr/bin/env python3
"""
Extract the 74 NeuroVFM diagnoses from GENERATED reports with Gemma via vLLM,
then emit a binary CSV restricted to the challenge's ranked label subset.

Reuses the prompt, diagnoses block, and JSON parsing from the handover script
(extract_neurovfm_dx_gemma.py) verbatim, so extraction matches how the ground
truth labels were produced (temperature=0, seed 42). Only the input side
differs: participant submissions carry one free-text report per study instead
of the sectioned MR-RATE CSVs.
"""
import argparse, csv, json
from pathlib import Path

from extract_neurovfm_dx_gemma import (
    NEUROVFM_PROMPT, build_diagnoses_block, build_user_message,
    parse_diagnosis_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True, type=Path)
    ap.add_argument("--diagnoses_json", required=True, type=Path)
    ap.add_argument("--keep_labels", required=True, type=Path,
                    help="newline-separated label subset used for scoring")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.pred_json, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw["generated_reports"] if isinstance(raw, dict) else raw
    studies = []
    for it in items:
        acc = Path(it["input_image_name"]).stem.replace(".nii", "")
        studies.append((acc, (it.get("report") or "").strip()))
    print(f"[extract] {len(studies)} generated reports loaded", flush=True)

    diagnoses = json.load(open(args.diagnoses_json))["diagnoses"]
    diag_keys = [d["key"] if isinstance(d, dict) else str(d) for d in diagnoses]
    preamble = NEUROVFM_PROMPT.format(diagnoses_block=build_diagnoses_block(diagnoses))
    keep = [l.strip() for l in open(args.keep_labels) if l.strip()]
    missing = [k for k in keep if k not in diag_keys]
    if missing:
        raise SystemExit(f"keep_labels not in diagnoses schema: {missing}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, dtype="bfloat16", seed=args.seed,
              max_model_len=12288, gpu_memory_utilization=0.93,
              # All 2029 prompts share the same ~9k-token preamble
              # (instructions + 74-diagnosis block); prefix caching
              # computes its KV once instead of per report — the
              # bulk of prefill disappears. Larger batches keep the
              # GPU saturated during decode.
              enable_prefix_caching=True,
              enforce_eager=False)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                              seed=args.seed)

    rows = []
    for start in range(0, len(studies), args.batch_size):
        chunk = studies[start:start + args.batch_size]
        convs = [[{"role": "user",
                   "content": build_user_message(preamble, text or "(empty report)")}]
                 for _, text in chunk]
        outs = llm.chat(convs, sampling, use_tqdm=False)
        for (acc, _), out in zip(chunk, outs):
            text = out.outputs[0].text
            parsed = parse_diagnosis_json(text, diag_keys)  # {key: (label, rationale)} | None
            if parsed is None:
                print(f"WARN {acc}: unparseable model output; defaulting to all-no",
                      flush=True)
                parsed = {}
            row = {"AccessionNo": acc}
            for k in keep:
                row[k] = int(parsed.get(k, (0, ""))[0])
            rows.append(row)
        print(f"[extract] {min(start+args.batch_size, len(studies))}/{len(studies)}",
              flush=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["AccessionNo"] + keep)
        w.writeheader()
        w.writerows(rows)
    print(f"[extract] wrote {args.out_csv} ({len(rows)} rows x {len(keep)} labels)",
          flush=True)


if __name__ == "__main__":
    main()
