```markdown
# VLM3D Challenge Docker Examples

This repository contains Dockerized pipelines and example containers for the VLM3D Challenge, covering both evaluation scripts for challenge submissions and model‐inference examples.

## Repository Structure

```

.
├── abnclass\_eval/
│   ├── Dockerfile
│   ├── test.sh           # verify abnormality classification evaluation
│   ├── export.sh         # package results for submission
│   └── README.md         # detailed instructions
├── ctgen\_eval/
│   ├── evaluation.py     # master script for FVD, CLIP, FID-2.5D
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── reportgen\_eval/
│   ├── evaluation.py     # master script for classification, CRG, NLG metrics
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── reportgen\_model\_example/
│   ├── Dockerfile
│   ├── ctchat\_pipeline.py
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── ct\_generation\_example/
│   ├── Dockerfile
│   ├── generate\_ct\_pipeline.py
│   ├── test.sh
│   ├── export.sh
│   └── README.md
├── classification\_example/
│   ├── Dockerfile
│   ├── ctclip\_classifier\_pipeline.py
│   ├── test.sh
│   ├── export.sh
│   └── README.md
└── README.md             # this file

````

## Supported Pipelines

1. **Abnormality Classification Evaluation** (`abnclass_eval/`)  
   Evaluate predicted probabilities against ground truth using AUROC, F1-Score and CRG-Score metrics.

2. **CT Generation Evaluation** (`ctgen_eval/`)  
   Compute FVD (CT-Net), CLIPScore, and 2.5-D FID on generated vs. reference 3D volumes.

3. **Report Generation Evaluation** (`reportgen_eval/`)  
   Aggregate multi-label classification, CRG metrics, and NLG metrics into a single `metrics.json`.

4. **Report Generation Model Example** (`reportgen_model_example/`)  
   Demonstrate CT→Report inference using a CTViT visual encoder + LLaVA-LLaMA3 CT-CHAT model.

5. **CT Synthesis Model Example** (`ct_generation_example/`)  
   Two-stage text-to-CT pipeline: MaskGIT transformer for low-res volumes, then diffusion super-resolution.

6. **Abnormality Classification Model Example** (`classification_example/`)  
   CT-CLIP pipeline with CTViT + Biomed-BERT + ImageLatentsClassifier for multi-label scoring.

## Quickstart

### Prerequisites

- Docker Engine (with NVIDIA runtime for GPU support)  
- NVIDIA drivers & CUDA (for GPU‐accelerated containers)

### Build & Test

For each pipeline or example folder:

```bash
cd <pipeline_folder>
docker build -t vlm3d-<pipeline_name> .
./test.sh
````

### Export for Submission

```bash
./export.sh
```

Each `export.sh` produces a `.tar.gz` or `predictions.zip` ready for upload to the VLM3D challenge platform.
```
```
