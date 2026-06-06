
# CT Generation Pipeline Docker

This Docker container generates 3D CT volumes from text prompts using a two-stage pipeline. You can use this to start creating your own model docker.

## Models

* **Low-Resolution Generation** – MaskGIT-based transformer sampling at low spatial resolution  
* **Super-Resolution** – diffusion-based UNet cascade to upscale low-res volumes to full resolution  

## Input format (text prompts at `/input/prompts.json`)

Unlike the abnclass/reportgen tracks, **ctgen does NOT receive CT volumes as input** — the platform mounts a single text file at `/input/prompts.json` (a JSON array of `{"input_image_name", "report"}` objects). Your container reads that file and writes generated `.nii.gz` volumes to `/output`. The `.nii.gz` input format that the abnclass / reportgen READMEs mention does not apply here.

`forithmus generate` will produce a sample `prompts.json` matching this phase's schema for local dry-runs.

## Build & test locally

Quick recap (see the [top-level README](../../README.md) for the full end-to-end flow including weights packaging and submission):

```bash
# from the repo root
docker build -t ctgen-thin:latest ct_challenges/ctgen_example_docker/
forithmus generate                            # writes a sample /input/prompts.json
forithmus test ctgen-thin:latest -t 1200      # generation is slow; bump --timeout
```

## Input Specification

Mount your prompts file to `/input/prompts.json`. The file must be a JSON array of objects with the following fields:
````markdown

  {
    "input_image_name": "<filename_without_extension>",
    "report": "<text_prompt>"
  },

````

**Notes:**

* `input_image_name` defines the base name for the output `.nii.gz` file.
* `report` is the radiology report text used as the generation prompt.

## Ground-Truth Data

Model checkpoints are baked into the container under:

```
/opt/app/models/ctvit_pretrained.pt
/opt/app/models/transformer_pretrained.pt
/opt/app/models/superres_pretrained.pt
```

## Output Specification

After generation, the container writes to `/output`:

* One `.nii.gz` volume per prompt, named `<input_image_name>.nii.gz`

## Testing

A test script is included to verify functionality. To run it:

```bash
./test.sh
```

Ensure that `test.sh` has execute permissions:

```bash
chmod +x test.sh
```

## Exporting

Use the provided `export.sh` script to package your results:

```bash
./export.sh
```

This produces a `.tar.gz` archive ready for submission.

*For questions or issues, please contact the maintainers.*\`\`\`
