# CT Generation Evaluation Docker

This Docker container evaluates CT volume generation predictions by running multiple metrics and merging their outputs into a single JSON file.

## Metrics

* **FVD\_CTNet** – Frechet Video Distance computed on 3D CT volumes using the CT-Net backbone.
* **CLIPScore / CLIP\_I2I** – CLIP-based image-text similarity, reporting both I2T and I2I scores and their mean.
* **FID\_2p5D** – 2.5D Frechet Inception Distance computed on orthogonal slices (XY, XZ, YZ).

## Input Specification

The container accepts either:

```bash
# Mount predictions directory or ZIP archive to /input
docker run --rm \
  -v "$(pwd)/input":/input \
  -v "$(pwd)/output":/output \
  forithmus/ctgen-eval:latest
```

Inside `/input`, provide either:

* A set of flattened `.mha` files under `/input` or any nested subdirectories.
* A single `.zip` archive anywhere under `/input` containing `.mha` files.

**Notes:**

* The first matching `.mha` files or the first `.zip` found will be used for evaluation.

## Ground-Truth Data

Ground-truth `.mha` volumes are baked into the container at:

```
/opt/app/ground-truth
```

Each file in this directory should be named by accession or unique identifier and have a `.mha` extension.

## Output Specification

After evaluation, the container writes the merged metrics JSON to `/output/metrics.json`. The file has the following structure:

```json
{
  "FVD_CTNet": <float>,        // FVD score
  "CLIPScore": <float>,        // CLIP I2T score
  "CLIPScore_I2I": <float>,    // CLIP I2I score
  "CLIPScore_mean": <float>,   // mean CLIP score
  "FID_2p5D_Avg": <float>,     // average 2.5D FID
  "FID_2p5D_XY": <float>,      // XY-plane FID
  "FID_2p5D_XZ": <float>,      // XZ-plane FID
  "FID_2p5D_YZ": <float>       // YZ-plane FID
}
```

All values are floats rounded to four decimal places.

## Testing

To verify functionality, run:

```bash
./test.sh
```

Ensure the script has execute permissions:

```bash
chmod +x test.sh
```

## Exporting

Use the `export.sh` script to set environment variables before running evaluation:

```bash
source ./export.sh
```

This will generate a `.tar.gz` file for submission to the challenge platform.

*For questions or issues, please contact the challenge organizers.*
