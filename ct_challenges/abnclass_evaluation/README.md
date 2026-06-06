# Abnormality Classification Evaluation Docker

This Docker container evaluates predicted abnormality classifications for CT volumes using standard metrics.

## Metrics

* **AUROC** – macro-average area under the ROC curve (threshold-independent separability).
* **F1-Score** – macro-average harmonic mean of precision and recall.
* **CRG-Score** – clinically-weighted relevance score (see challenge guidelines for weight definitions).

## Input format (`.nii.gz` from the platform)

The participant's algorithm container receives the per-phase test set as `*.nii.gz` volumes under `/input/`, and writes `results.json` to `/output/`. On the Forithmus platform, this eval container then reads those predictions from `/input/predictions/results.json` (mounted read-only from the participant's submission output) and the hidden ground truth from `/input/ground_truth/` (mounted read-only from a bucket only the per-challenge eval service account can read). It writes `/output/metrics.json` to local disk; the trampoline uploads it to GCS after the eval process exits — see top-level README §6.

## Build & test locally

Quick recap (see the [top-level README](../../README.md) for the full end-to-end flow):

```bash
# from the repo root
docker build -t abnclass-eval:latest ct_challenges/abnclass_evaluation/
./ct_challenges/abnclass_evaluation/test.sh    # runs against the bundled smoke-test predictions + GT
```

For host upload to the platform: `forithmus upload-eval abnclass-eval.tar.gz --phase <phase>`.

## Input Specification

The platform mounts two read-only directories (the eval reads `/input/predictions/`, NOT a baked-in file):

```text
/input/
  predictions/    # the participant container's /output/results.json, wrapped by the
                  # algorithm's entrypoint into the Grand-Challenge schema:
                  #   [{"outputs":[{"value":{"predictions":[ ... ]}}]}]
  ground_truth/   # ground_truth.csv (hidden; eval-SA only)
/output/
  metrics.json
```

Each entry inside `predictions` has the shape:

```json
{
  "input_image_name": "<case id, extension stripped>",
  "probabilities": { "Medical material": <0–1>, "Cardiomegaly": <0–1>, "...": "all 18 labels" }
}
```

**Notes:**

* `input_image_name` must match the ground-truth IDs (the case id with any volume extension — `.nii.gz` / `.nii` / `.mha` — stripped).
* Probabilities must be floats between 0 and 1.

## Ground-Truth Data

Ground truth is **not** baked into the image — the platform mounts `ground_truth.csv` at `/input/ground_truth/ground_truth.csv` (readable only by the per-challenge eval service account). The CSV has columns:

```
AccessionNo,<label_1>,<label_2>,...
```

with binary values (0 or 1) for each of the 18 labels.

## Output Specification

After evaluation, the container writes metrics to `/output/metrics.json`. The JSON has two top-level sections:

1. **`crg`** – clinically-weighted relevance metrics and counts:
   ```json
   "crg": {
     "A": <float>,          // unnormalized A component
     "U": <float>,          // unnormalized U component
     "X": <float>,          // unnormalized X component
     "r": <float>,          // ratio r = X/U
     "FN": <int>,           // false negatives count
     "FP": <int>,           // false positives count
     "TP": <int>,           // true positives count
     "CRG": <float>,        // final CRG-Score (0–1)
     "score_s": <float>     // scaled score for ranking
   }
````

2. **`classification`** – macro and per-pathology classification metrics:

   ```json
   "classification": {
     "macro": {
       "f1": <float>,        // macro F1-Score
       "auroc": <float>,     // macro AUROC
       "recall": <float>,    // macro recall
       "accuracy": <float>,  // macro accuracy
       "precision": <float>  // macro precision
     },
     "per_pathology": [
       {
         "name": "<label>",
         "f1": <float>,
         "auroc": <float>,
         "recall": <float>,
         "accuracy": <float>,
         "precision": <float>
       },
       ...
     ]
   }
   ```

The top-level `"crg"` → `"CRG"` field is the overall CRG-Score (the primary ranking metric), `"classification"` → `"macro"` → `"auroc"` is the macro AUROC, and `"classification"` → `"macro"` → `"f1"` is the macro F1-Score.


## Testing

A test script is included to verify functionality. To run the tests:

```bash
./test.sh
```

Ensure that `test.sh` has execute permissions:

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
