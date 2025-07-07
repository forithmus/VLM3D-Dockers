# VLM3D Challenge Docker Examples

This repository provides Dockerized pipelines and example containers for the VLM3D Challenge. It includes evaluation scripts for challenge submissions and reference implementations for model inference.

## Repository Structure

```text
.
├── abnclass_eval/
│   ├── Dockerfile
│   ├── test.sh           # Verify abnormality classification evaluation
│   ├── export.sh         # Package results for submission
│   └── README.md         # Detailed instructions
├── ctgen_eval/
│   ├── evaluation.py     # Master script for FVD, CLIPScore, and 2.5D FID
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── reportgen_eval/
│   ├── evaluation.py     # Master script for classification, CRG, and NLG metrics
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── reportgen_model_example/
│   ├── Dockerfile
│   ├── ctchat_pipeline.py
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── ct_generation_example/
│   ├── Dockerfile
│   ├── generate_ct_pipeline.py
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── classification_example/
│   ├── Dockerfile
│   ├── ctclip_classifier_pipeline.py
│   ├── test.sh
│   ├── export.sh
│   └── README.md
└── README.md             # This file
```

## Supported Pipelines

* **Abnormality Classification Evaluation** (`abnclass_eval/`)
  Evaluate predicted probabilities against ground truth using AUROC, F1-Score, and CRG-Score.

* **CT Generation Evaluation** (`ctgen_eval/`)
  Compute FVD (CT-Net), CLIPScore, and 2.5-D FID on generated versus reference 3D volumes.

* **Report Generation Evaluation** (`reportgen_eval/`)
  Aggregate multi-label classification, CRG, and NLG metrics into a single `metrics.json`.

* **Report Generation Model Example** (`reportgen_model_example/`)
  Demonstrate CT→Report inference using a CTViT visual encoder and CT-CHAT (LLaVA-LLaMA3).

* **CT Synthesis Model Example** (`ct_generation_example/`)
  Two-stage text-to-CT pipeline: low-resolution MaskGIT transformer followed by diffusion-based super-resolution.

* **Abnormality Classification Model Example** (`classification_example/`)
  Multi-label scoring pipeline using CTViT, Biomed-BERT, and ImageLatentsClassifier.

## Quickstart

### Prerequisites

* Docker Engine (with NVIDIA Container Toolkit for GPU support)
* NVIDIA drivers and CUDA toolkit installed

### Building and Testing

For each example or evaluation folder:

```bash
cd <folder_name>
docker build -t vlm3d-<folder_name> .
./test.sh
```

### Exporting for Submission

```bash
./export.sh
```

Each `export.sh` produces a `.tar.gz` or `predictions.zip` package ready for upload to the VLM3D challenge platform.")
