````markdown
# CT Generation Pipeline Docker

This Docker container generates 3D CT volumes from text prompts using a two-stage pipeline. You can use this to start creating your own model docker.

## Metrics

* **Low-Resolution Generation** – MaskGIT-based transformer sampling at low spatial resolution  
* **Super-Resolution** – diffusion-based UNet cascade to upscale low-res volumes to full resolution  

## Input Specification

Mount your prompts file to `/input/prompts.json`. The file must be a JSON array of objects with the following fields:

```json
[
  {
    "input_image_name": "<filename_without_extension>",
    "report": "<text_prompt>"
  },
  ...
]
````

**Notes:**

* `input_image_name` defines the base name for the output `.mha` file.
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

* One `.mha` file per prompt, named `<input_image_name>.mha`
* A ZIP archive `predictions.zip` containing all generated `.mha` files

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
