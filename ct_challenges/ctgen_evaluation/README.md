# CT Generation Evaluation Docker

This Docker container evaluates CT volume generation predictions by running multiple metrics and merging their outputs into a single JSON file.

## Metrics

* **FVD\_CTNet** – Frechet Video Distance computed on 3D CT volumes using the CT-Net backbone.
* **CLIPScore / CLIP\_I2I** – CLIP-based image-text similarity, reporting both I2T and I2I scores and their mean.
* **FID\_2p5D** – 2.5D Frechet Inception Distance computed on orthogonal slices (XY, XZ, YZ).

## Input format (predictions + ground truth)

ctgen is the one VLM3D track whose algorithm-side input is **not** a volume — the algorithm receives text prompts and writes generated volumes. The algorithm writes loose **`.nii.gz`** volumes (one per prompt, named by `input_image_name`); this eval container sees them at `/input/predictions/` (read-only from the participant's submission output; loose `.mha` and a `predictions.zip` are still accepted for backward compatibility) and the held-out real CT volumes at `/input/ground_truth/` (read-only from the per-phase ground-truth bucket; eval-SA only). Predictions are matched to ground truth by **case stem** (extension-agnostic), so a `.nii.gz` prediction matches an `.mha` GT. It writes `/output/metrics.json` to local disk; the trampoline uploads it to GCS after the eval exits — see top-level README §6.

## Build & test locally

Quick recap (see the [top-level README](../../README.md) for the full end-to-end flow):

```bash
# from the repo root
docker build -t ctgen-eval:latest ct_challenges/ctgen_evaluation/
./ct_challenges/ctgen_evaluation/test.sh       # smoke-test against the bundled fixtures
```

For host upload to the platform: `forithmus upload-eval ctgen-eval.tar.gz --phase <phase>`.

## Input Specification

The platform mounts two read-only directories:

```text
/input/
  predictions/    # participant's generated volumes (*.nii.gz; .mha / predictions.zip also accepted)
  ground_truth/   # held-out real CT volumes (*.mha / *.nii.gz), eval-SA only
/output/
  metrics.json    # written here
```

Predictions are matched to ground truth by case stem (the filename with its extension stripped), so extensions need not match between the two.

## Output Specification

After evaluation, the container writes the merged metrics to `/output/metrics.json`, wrapped under a top-level `metrics` key (the leaderboard reads `$.metrics.FVD_CTNet`):

```json
{
  "metrics": {
    "FVD_CTNet": <float>,        // FVD score (primary ranking metric)
    "CLIPScore": <float>,        // CLIP image↔text score
    "CLIPScore_I2I": <float>,    // CLIP image↔image score
    "CLIPScore_mean": <float>,   // mean CLIP score
    "FID_2p5D_Avg": <float>,     // average 2.5D FID
    "FID_2p5D_XY": <float>,      // XY-plane FID
    "FID_2p5D_XZ": <float>,      // XZ-plane FID
    "FID_2p5D_YZ": <float>       // YZ-plane FID
  }
}
```

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

`export.sh` saves the built image as a `.tar.gz` for host upload:

```bash
./export.sh                  # docker save ctgen-eval | gzip > ctgen-eval.tar.gz
forithmus upload-eval ctgen-eval.tar.gz --phase <phase>
```

*For questions or issues, please contact the challenge organizers.*
