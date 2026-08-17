# MR Volume Generation — Example Algorithm Container

Reference baseline for the `mr-volume-generation` track: a MAISI-style latent
diffusion generator (NVIDIA NV-Generate-CTMR, vendored under
`NV-Generate-CTMR/` — demo assets and tutorial notebooks trimmed) conditioned
on report text via `microsoft/BiomedVLP-CXR-BERT-specialized`.

## I/O contract

- Input: report prompts; each entry's `input_image_name` names the target.
- Output: one `/output/{input_image_name}.nii.gz` volume per entry — the
  filename must match its ground-truth target
  (`{study_uid}_{modality}-raw-{plane}.nii.gz`); scoring parses the modality
  from the name (see `../mrgen_evaluation/`).
- Weights ship separately (`forithmus submit --weights weights.zip`); the
  extracted bundle is read from the platform weights dir.

Note: this baseline writes progress to `/checkpoint` and handles SIGTERM, so
spot preemptions and timeouts resume instead of restarting — copy that
pattern for your own long-running submissions.

## Build & submit

```bash
docker build -t mr-gen-baseline .
forithmus init mr-volume-generation
forithmus test mr-gen-baseline --timeout 1200
docker save mr-gen-baseline | gzip > image.tar.gz
forithmus submit image.tar.gz --tier gpu-a100-40 --time-budget 480 --weights weights.zip
```
