# BTB3D Report Generation Docker (reportgen baseline)

This is the **deployed reportgen baseline** for the VLM3D challenge — the exact thin container that ran in production and scored **CRG ≈ 0.346**. It generates a free-text radiology report for each chest CT volume using a MAGViT-2 visual tokenizer and a LoRA-fine-tuned Llama-3.1-8B language model.

Use it as a working reference for building your own report-generation submission. It is a **thin image**: model weights are *not* baked in — they are uploaded separately as `weights.zip` and mounted at `/weights` at runtime (see the [top-level README](../../README.md) for the full weights-packaging + submission flow).

## Model

* **MAGViT-2 encoder** — tokenizes the 3D CT volume into discrete visual tokens (`3rd_stage.ckpt`).
* **LLaVA Llama-3.1-8B + LoRA** — consumes the visual tokens and decodes a radiology report (`Llama-3.1-8B-Instruct` base + `checkpoint-43000` LoRA adapter).

Quantization is a build flag: `--build-arg BTB3D_QUANT=bf16` (default, A100/L4) or `--build-arg BTB3D_QUANT=4bit` (T4-friendly).

## Input format (`.nii.gz` from the platform)

The platform mounts the per-phase test set at `/input/` as compressed NIfTI (`*.nii.gz`), one CT volume per case (recursively discovered). The filename pattern is `{batch}_{patient}_{series}.nii.gz`; `predict.py` derives `input_image_name` as the **series UUID** (the bare filename with the `.nii.gz` suffix stripped). Use `forithmus generate` to materialize a sample `/input` tree for local testing.

## Runtime layout

```text
/input/                         # *.nii.gz CT volumes (mounted by the platform)
/output/predictions.json        # written by predict.py (see schema below)
/weights/                       # extracted from your weights.zip at runtime
  3rd_stage.ckpt                # MAGViT-2 encoder checkpoint
  Llama-3.1-8B-Instruct/        # base LLM
  checkpoint-43000/             # LoRA adapter
/checkpoint/state.json          # resumable progress (written after each volume)
```

`WEIGHTS_DIR` defaults to `/weights` in the Dockerfile, so the contents of your uploaded `weights.zip` land exactly where `predict.py` expects them.

## Output schema

`predict.py` writes the **Grand-Challenge-wrapped** schema the production eval container consumes:

```json
[
  {
    "outputs": [
      {
        "value": {
          "name": "Generated reports",
          "type": "Report generation",
          "version": "1.0",
          "generated_reports": [
            { "input_image_name": "<series-uuid>", "report": "<generated report text>" }
          ]
        }
      }
    ]
  }
]
```

## Fail-loud inference

The per-volume inference loop in `predict.py` runs **without a `try/except` wrapper on purpose**. If a volume OOMs or otherwise fails, the container crashes immediately so the run fails fast — rather than silently skipping volumes and "succeeding" with partial predictions that burn the whole compute budget. (The container *does* checkpoint completed volumes to `/checkpoint/state.json` so a SIGTERM-interrupted run can resume, but a hard error is never swallowed.)

## Build & test locally

See the [top-level README](../../README.md) for the full end-to-end flow (weights packaging, submission, watching the score). Quick local smoke test:

```bash
# from this directory
./build.sh                       # docker build -t reportgen-btb3d .
# drop your weights under ./weights/ (3rd_stage.ckpt, Llama-3.1-8B-Instruct/, checkpoint-43000/)
./test.sh                        # runs the container on ./test, prints /output/predictions.json
```

LLM decode is slow — when submitting, give the run a generous `--time-budget` (e.g. `forithmus submit ... -t 1200`).

## Export for submission

```bash
./export.sh                      # docker save reportgen-btb3d | gzip > reportgen-btb3d.tar.gz
forithmus submit reportgen-btb3d.tar.gz --phase <phase> --tier gpu-l4-xl --weights weights.zip -t 1200 -d "btb3d v1"
```

*For questions or issues, please contact the maintainers.*
