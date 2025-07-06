# Abnormality Classification Evaluation Docker

This Docker container evaluates predicted abnormality classifications for CT volumes using standard metrics.

## Metrics

* **AUROC** – macro-average area under the ROC curve (threshold-independent separability).
* **F1-Score** – macro-average harmonic mean of precision and recall.
* **CRG-Score** – clinically-weighted relevance score (see challenge guidelines for weight definitions).

## Input Specification

Mount your predictions file to `/input/predictions.json`. The file must be a JSON array of objects with the following fields:

```json
[
  {
    "input_image_name": "<filename_without_extension>",
    "probabilities": {
      "abnormality_label_1": <probability_0–1>,
      "abnormality_label_2": <probability_0–1>,
      ...
    }
  },
  ...
]
```

**Notes:**

* `input_image_name` must exactly match the ground-truth IDs (filenames without `.mha`).
* Probabilities must be floats between 0 and 1.

## Ground-Truth Data

Ground-truth labels are baked into the container at:

```
/opt/app/ground-truth/ground_truth.csv
```

This CSV has columns:

```
AccessionNo,abnormality_label_1,abnormality_label_2,...
```

with binary values (0 or 1) for each label.

````markdown
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

All floats are rounded to four decimal places. The top-level `"crg"` → `"CRG"` field corresponds to the overall CRG-Score, `"classification"` → `"macro"` → `"auroc"` to the macro AUROC, and `"classification"` → `"macro"` → `"f1"` to the macro F1-Score.\`\`\`


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

You can use the provided `export.sh` script to set environment variables before running:

```bash
./export.sh
```
This will generate a .tar.gz file that you will upload to the challenge platform.

*For questions or issues, please contact the challenge organizers.*
