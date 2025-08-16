# Abnormality Localization Evaluation Docker

This Docker container evaluates predicted abnormality localizations for CT volumes using bounding box–based metrics.

## Metrics

* **IoU (Intersection-over-Union)** – macro-average 3D overlap between predicted and ground-truth boxes.
* **Distance** – macro-average centroid distance (mm) between matched boxes.
* **FROC** – sensitivity at fixed false-positive budgets (0.5, 1, 2, 4 FP per scan).

## Input Specification

Mount your predictions file to `/input/predictions.json`. The file must be a JSON array with the following structure:

```json
[
  {
    "outputs": [
      {
        "type": "predictions",
        "value": {
          "predictions": [
            {
              "input_image_name": "<filename_without_extension>.mha",
              "Pericardial effusion": [
                {
                  "bbox_mm": [x_min, y_min, z_min, dx, dy, dz],
                  "probability": <float_0–1>
                }
              ],
              "Pleural effusion": [],
              "Consolidation": [],
              "Ground glass opacity": [],
              "Lung nodule": []
            }
          ]
        }
      }
    ]
  }
]
```

**Notes:**

* `input_image_name` must exactly match the ground-truth IDs (filenames without `.mha`).
* Each pathology key must exist, even if empty.
* Each bounding box must contain exactly 6 numbers: `[x_min, y_min, z_min, dx, dy, dz]`.
* `probability` is optional (defaults to 1.0).

## Ground-Truth Data

Ground-truth bounding boxes are baked into the container at:

```
/opt/app/ground-truth/bbox_ground_truth.csv
```

This CSV has columns:

```
id,pericardial_effusion,pleural_effusion,consolidation,ggd,nodule
```

Each cell contains 0, 1, or multiple bounding boxes in `[x_min, y_min, z_min, dx, dy, dz]` format.

## Output Specification

After evaluation, the container writes metrics to `/output/metrics.json`. The JSON has three top-level sections:

1. **`iou`** – per-pathology and macro Intersection-over-Union:

   ```json
   "iou": {
     "per_pathology": {
       "Pericardial effusion": <float>,
       "Pleural effusion": <float>,
       ...
     },
     "macro": <float>
   }
   ```

2. **`distance`** – per-pathology and macro centroid distances (mm):

   ```json
   "distance": {
     "per_pathology": {
       "Pericardial effusion": <float>,
       "Pleural effusion": <float>,
       ...
     },
     "macro": <float>
   }
   ```

3. **`froc`** – sensitivity at fixed false-positive budgets:

   ```json
   "froc": {
     "05_fp_per_image": <float>,
     "1_fp_per_image": <float>,
     "2_fp_per_image": <float>,
     "4_fp_per_image": <float>
   }
   ```

All floats are rounded to four decimal places.

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

This will generate a `.tar.gz` file that you will upload to the challenge platform.

*For questions or issues, please contact the challenge organizers.*
