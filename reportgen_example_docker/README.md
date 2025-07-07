# CT-CHAT Report Generation Docker

This Docker container generates radiology reports for chest CT volumes using a two-stage pipeline: a 3D CTViT visual encoder and a LLaVA-based LLaMA3 model fine-tuned as CT-CHAT. You can use this Docker to create your own model container for submission.

## Metrics

* **CTViT Visual Encoder** – patch-based 3D transformer (512‑dim codebook) that produces volume embeddings
* **CT-CHAT Report Generator** – LLaVA LLaMA3 1.8B model fine-tuned with LoRA for radiology report synthesis

## Input Specification

Mount a directory of CT volumes to `/input`. Supported extensions: `.mha`, `.nii`, `.nii.gz`. Each file is processed independently; the base filename (without extension) is used as `input_image_name`.

```text
/input/
  input1.mha
  …
```

## Ground-Truth Data

Model weights and checkpoints baked into the container at:

```text
/opt/app/models/clip_visual_encoder.pth
/opt/app/models/llava-llama3_1_8B_ctclip-finetune_256-lora_2gpus
/opt/app/llama             # base LLaMA installation
```

## Output Specification

After running, the container writes its JSON output to `/output/results.json`:

```json
{
  "name": "Generated reports",
  "type": "Report generation",
  "generated_reports": [
    {
      "input_image_name": "<filename_without_extension>",
      "report": "<generated_report_text>"
    }
    // one entry per volume
  ],
  "version": {
    "major": 1,
    "minor": 0
  }
}
```

## Testing

A test script is included to verify the pipeline. To run the tests:

```bash
./test.sh
```

Ensure `test.sh` is executable:

```bash
chmod +x test.sh
```

## Exporting

Use the provided `export.sh` script to set any required environment variables and package your results:

```bash
./export.sh
```

This will create a `.tar.gz` archive ready for submission.

*For questions or issues, please contact the maintainers.*
