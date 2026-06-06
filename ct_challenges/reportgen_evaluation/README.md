# Report Generation Evaluation Docker

This Docker container evaluates radiology report generation by running inference, classification scoring, clinically-weighted relevance (CRG), and natural language generation (NLG) metrics, then merges all results into a single JSON file.

## Metrics

* **NLG** – natural language generation metrics (e.g., BLEU, ROUGE, METEOR).
* **Classification** – multi-label classification metrics on inferred labels (macro F1, AUROC, recall, accuracy, precision).
* **CRG-Score** – clinically-weighted relevance score (see challenge guidelines for weight definitions).

## Input format (`.nii.gz` from the platform)

The participant's algorithm container receives the per-phase test set as `*.nii.gz` CT volumes under `/input/` and writes a `predictions.json` to `/output/`. On the Forithmus platform, this eval container then reads those predictions from `/input/predictions/` (read-only from the submission output) and the hidden ground truth from `/input/ground_truth/` (read-only, eval-SA only — **no longer baked into the image** as older Grand-Challenge variants did). It writes `/output/metrics.json` to local disk; the trampoline uploads it to GCS after eval exits — see top-level README §6.

The production eval expects the Grand-Challenge-wrapped predictions schema `[{"outputs":[{"value":{...,"version":"1.0",...}}]}]`. The simpler `[{"input_image_name","report","labels"}]` shape documented below corresponds to an older eval contract that read flat JSON; the production pipeline now uses the wrapped form.

> **Asset note.** This eval image needs `RadBertClassifier.pth` (~498 MB) and `radbert_local/` (~479 MB) for the RadBERT label-inference head. Those binaries are NOT checked into this repo (Git LFS quota). Pull them from the challenge's release-asset/GCS path before `docker build`, or this image will fail at runtime.

## Build & test locally

Quick recap (see the [top-level README](../../README.md) for the full end-to-end flow):

```bash
# from the repo root
# (first: fetch RadBertClassifier.pth + radbert_local/ into reportgen_evaluation/assets/)
docker build -t reportgen-eval:latest ct_challenges/reportgen_evaluation/
./ct_challenges/reportgen_evaluation/test.sh    # smoke-test against the bundled fixtures
```

For host upload to the platform: `forithmus upload-eval reportgen-eval.tar.gz --phase <phase>`.

## Input Specification

The platform mounts two read-only directories:

```text
/input/
  predictions/    # the participant's /output/predictions.json (Grand-Challenge-wrapped)
  ground_truth/   # ground_truth.json + ground_truth.csv (hidden; eval-SA only)
/output/
  metrics.json
```

`predictions.json` is the Grand-Challenge-wrapped form the algorithm's `predict.py` writes:

```json
[{"outputs": [{"value": {
  "name": "Generated reports",
  "type": "Report generation",
  "version": "1.0",
  "generated_reports": [
    { "input_image_name": "<case id>", "report": "<generated report text>" }
  ]
}}]}]
```

**Notes:**

* `input_image_name` must match the ground-truth identifiers (case id with the volume extension — `.nii.gz` / `.nii` / `.mha` — stripped). Labels are inferred from the generated text by the eval's RadBERT head, so participants only supply `input_image_name` + `report`.

## Ground-Truth Data

Ground truth is **not** baked into the image — the platform mounts it at `/input/ground_truth/` (readable only by the per-challenge eval service account):

* `ground_truth.json` — reference reports: array of `{ "input_image_name", "report" }`.
* `ground_truth.csv` — reference labels: `AccessionNo,<label_1>,<label_2>,...` with binary values, used by the classification + CRG scorers.

## Output Specification

After evaluation, the container writes metrics to `/output/metrics.json`. The JSON has three top-level sections:

1. **`generation`** – NLG metrics object, e.g.:

   ```json
   "generation": {
     "BLEU": <float>,
     "ROUGE_L": <float>,
     "METEOR": <float>,
     ...
   }
   ```

2. **`classification`** – classification metrics object:

   ```json
   "classification": {
     "macro": {
       "f1": <float>,
       "auroc": <float>,
       "recall": <float>,
       "accuracy": <float>,
       "precision": <float>
     }
   }
   ```

3. **`crg`** – clinically-weighted relevance metrics:

   ```json
   "crg": {
     "A": <float>,
     "U": <float>,
     "X": <float>,
     "r": <float>,
     "FN": <int>,
     "FP": <int>,
     "TP": <int>,
     "CRG": <float>,
     "score_s": <float>
   }
   ```

The final merged JSON has these sections; `crg.CRG` is the primary ranking metric.

## Testing

To verify functionality, run:

```bash
./test.sh
```

Ensure the script is executable:

```bash
chmod +x test.sh
```

## Exporting

`export.sh` saves the built image as a `.tar.gz` for host upload:

```bash
./export.sh
forithmus upload-eval <image>.tar.gz --phase <phase>
```

*For questions or issues, please contact the challenge organizers.*
