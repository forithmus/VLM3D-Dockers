# MR Abnormality Classification — Evaluation Container

The production scoring container for `mr-abnormality-classification`.

Contract (set up by the platform):

```
/input/predictions/    first *.json = participant submission
/input/ground_truth/   ground_truth.csv (32 binary labels per study)
/output/metrics.json   macro + per-label precision/recall/F1/accuracy/AUROC
```

Primary leaderboard metric: **macro AUROC**. Build with
`docker build -t mr-abnclass-eval .` and run against the contract above to
reproduce scoring locally.
