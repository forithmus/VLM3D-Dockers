# CT-CLIP Abnormality Classification Docker

This Docker container performs multi-label abnormality classification on chest CT volumes using:

* **CTViT Visual Encoder**: Patch-based 3D vision transformer producing volume embeddings
* **Biomed-BERT Text Encoder**: Biomedical BERT tokenizer and encoder for text features
* **CT-CLIP Classifier**: Linear classifier on image latents for multi-label probability scoring

---

## Directory Layout

```text
input/                   ← Directory of CT volumes (*.mha) mounted at runtime
output/                  ← Directory where `results.json` will be written
/opt/app/models/         ← Preloaded model weights and checkpoints
    ├─ clip_visual_encoder.pth
    ├─ BiomedVLP-CXR-BERT-specialized/
    └─ ctclip_classifier.pth
test.sh                  ← Script to validate container functionality
export.sh                ← Script to package results for submission
Dockerfile               ← Builds the container image
```

---

## Quickstart

### 1. Build the Image

```bash
docker build -t vlm3d-classification .
```

### 2. Run Inference

Mount your volumes folder to `/input` and an empty folder to `/output`:

```bash
docker run --rm \
  --gpus all \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  vlm3d-classification
```

Each `.mha` file in `/input` is processed independently. The filename (without extension) is used as `input_image_name`.

---

## Input Specification

* **Supported Formats**: `.mha`
* **Mount Point**: `/input`
* **Processing**: Each volume is loaded, preprocessed, and run through the CTViT-BERT-Classifier pipeline.

```text
/input/
  ├─ scan1.mha
  ├─ scan2.mha
  └─ ...
```

---

## Model Weights

The following files are baked into the container at build time:

```text
/opt/app/models/clip_visual_encoder.pth
/opt/app/models/BiomedVLP-CXR-BERT-specialized/  # tokenizer + model directory
/opt/app/models/ctclip_classifier.pth
```

---

## Output Specification

After processing, the container writes `results.json` to `/output`:

```json
{
  "name": "Generated probabilities",
  "type": "Abnormality Classification",
  "version": { "major": 1, "minor": 0 },
  "predictions": [
    {
      "input_image_name": "scan1",
      "probabilities": {
        "Medical material": 0.00,
        "Arterial wall calcification": 0.42,
        ...
      }
    },
    ...
  ]
}
```

* One entry per input volume
* Probabilities correspond to predefined abnormality labels

---

## Testing

Verify that the pipeline runs correctly on sample data:

```bash
chmod +x test.sh
./test.sh
```

---

## Exporting Results

Package the output for submission:

```bash
chmod +x export.sh
./export.sh
```

This script creates a `.tar.gz` archive containing `results.json`.

---

*For questions or issues, please contact the repository maintainers.*
