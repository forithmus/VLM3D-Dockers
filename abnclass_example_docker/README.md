# CT-CLIP Abnormality Classification Docker

This Docker container performs abnormality-classification inference on chest CT volumes using a CTViT visual encoder, a Biomed-BERT text encoder, and an ImageLatents classifier.

## Metrics

* **CTViT Visual Encoder** – patch-based 3D transformer produces volume embeddings
* **Biomed-BERT Text Encoder** – biomedical BERT tokenizer and encoder
* **CT-CLIP Classifier** – linear classifier on image latents for multi-label abnormality scoring

## Input Specification

Mount your CT volumes directory to `/input`. Supported file extensions:

```
.mha
```

Each volume is processed independently; the base filename (without extension) is used as `input_image_name`.

**Example:**

```text
/input/
  input1.mha
  …
```

## Ground-Truth Data

Model weights and checkpoints are baked into the container at:

```
/opt/app/models/clip_visual_encoder.pth
/opt/app/models/BiomedVLP-CXR-BERT-specialized
/opt/app/models/ctclip_classifier.pth
```

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

Use the provided `export.sh` script to package your results for submission:

```bash
./export.sh
```

This generates a `.tar.gz` archive containing `results.json`.

*For questions or issues, please contact the maintainers.*
