# MR Report Generation — Evaluation Container

The production scoring container for `mr-report-generation`. Three stages
(`src/evaluation.py` orchestrates):

1. **NLG metrics** — BLEU 1-4/mean, ROUGE-L, METEOR, CIDEr against the
   reference reports.
2. **Label extraction** — a local Gemma model (vLLM, temperature 0, fixed
   seed — the same pipeline that produced the ground-truth labels) extracts
   74 NeuroVFM diagnoses from each generated report; scoring is restricted
   to the 32-label challenge subset (`src/keep32.txt`).
3. **Classification + CRG** — per-label scores and the **CRG score**
   (primary leaderboard metric).

Contract (set up by the platform):

```
/input/predictions/    first *.json = participant submission
/input/ground_truth/   ground_truth.json, ground_truth.csv, _assets/gemma/
/output/metrics.json   {"generation": ..., "classification": ..., "crg": ...}
```

Notes: the extraction model (~59 GB bf16) is staged from the GT assets to
local disk before vLLM starts (mmap over FUSE is unusably slow); the eval
runs on `gpu-a100-80` and takes several hours for a full 2,029-report set.
