# CT-CLIP Abnormality Classification Docker

This Docker container performs abnormality-classification inference on chest CT volumes using a CTViT visual encoder, a Biomed-BERT text encoder, and an ImageLatents classifier.

## Models

* **CTViT Visual Encoder** – patch-based 3D transformer produces volume embeddings
* **Biomed-BERT Text Encoder** – biomedical BERT tokenizer and encoder
* **CT-CLIP Classifier** – linear classifier on image latents for multi-label abnormality scoring

## Input format (`.nii.gz` from the platform)

The Forithmus platform mounts the per-phase test set into your container at `/input/` as compressed NIfTI files (`*.nii.gz`) — one volume per case. Each volume is processed independently; the base filename with the `.nii.gz` suffix stripped is used as `input_image_name` in the output JSON.

You do not need to bake test data into the image — `forithmus generate` will create a sample `/input` tree shaped to this phase's schema for local testing.

## Build & test locally

Quick recap (see the [top-level README](../../README.md) for the full end-to-end flow including weights packaging and submission):

```bash
# from the repo root
docker build -t abnclass-thin:latest ct_challenges/abnclass_example_docker/
forithmus generate                            # writes a sample /input tree
forithmus test abnclass-thin:latest -t 600    # runs the container + validates output schema
```

## Input Specification

Mount your CT volumes directory to `/input`. The platform ships volumes as `.nii.gz`:

```
.nii.gz
```

Each volume is processed independently; the base filename (with the `.nii.gz` suffix stripped) is used as `input_image_name`.

**Example:**

```text
/input/
  input1.nii.gz
  …
```

## Model weights (thin image)

This is a **thin image**: model weights are *not* baked in. You upload them separately as `weights.zip` (see the [top-level README](../../README.md) for the packaging + submission flow), the platform extracts them to `/weights/` at runtime, and `entrypoint.sh` symlinks each into `/opt/app/models/` where `process.py` loads them:

```text
/weights/BiomedVLP-CXR-BERT-specialized   ->  /opt/app/models/BiomedVLP-CXR-BERT-specialized
/weights/CT_LiPro_v2.pt                    ->  /opt/app/models/CT_LiPro_v2.pt
```

So `weights.zip` must contain `BiomedVLP-CXR-BERT-specialized/` and `CT_LiPro_v2.pt` at its **root** (no parent directory). For a local test, drop the same files under `./weights/` and mount it at `/weights`.

## Output Specification

After inference, the container writes a single JSON file to `/output/results.json`:

```json
{
  "name": "Generated probabilities",
  "type": "Abnormality Classification",
  "version": {"major": 1, "minor": 0},
  "predictions": [
    {
      "input_image_name": "<filename_without_extension>",
      "probabilities": {
        "Medical material": 0.00,
        "Arterial wall calcification": 0.42,
        …
      }
    }
    // one entry per input volume
  ]
}
```

## Testing

A test script is included to verify functionality. To run the tests:

```bash
./test.sh
```

Ensure `test.sh` is executable:

```bash
chmod +x test.sh
```

## Exporting

`export.sh` saves the built Docker image as `abnclass-thin.tar.gz` for submission:

```bash
./export.sh
forithmus submit abnclass-thin.tar.gz --phase <phase> --tier gpu-l4-xl --weights weights.zip -d "my algorithm"
```

*For questions or issues, please contact the maintainers.*
