# MR Report Generation — Example Algorithm Container

Reference baseline for the `mr-report-generation` track: a V-JEPA2 video encoder
(MIL over the study's atlas-space series) feeding a MedGemma-4B report writer
(MR-RATE pipeline). Reads brain MR studies from `/input`, writes
`/output/predictions.json`.

## Input contract (important)

`/input` holds **one un-extracted zip per study** on a cloud-storage FUSE
mount: `<STUDY>.zip` containing `<STUDY>/{img,atlas,seg}/*.nii.gz`
(native series, MNI152-1mm-registered series, brain/defacing masks).
`run_inference_common.py` shows the recommended pattern: **lazy per-study
extraction with background prefetch** — the extracted dataset (~350 GB) does
not fit the 200 GB job disk, so never extract everything up front.

## Output

```json
{"generated_reports": [
  {"input_image_name": "<STUDY>.nii.gz", "report": "Free-text MR report ..."}
]}
```

## Build & submit

```bash
docker build -t mr-reportgen-baseline .
forithmus init mr-report-generation
forithmus test mr-reportgen-baseline --timeout 1200
docker save mr-reportgen-baseline | gzip > image.tar.gz
forithmus submit image.tar.gz --tier gpu-a100-80 --time-budget 1320 --weights weights.zip
```

Weights (V-JEPA2 backbone, encoder+writer checkpoints, MedGemma) ship as a
separate `weights.zip`, mounted extracted at `$FORITHMUS_WEIGHTS_DIR`
(see `entrypoint_rg.sh` for the expected layout). Contact the organizers via
the challenge page for the baseline weight bundle.

Throughput on `gpu-a100-80`: ~37 s/study (~21 h for the full 2,029-study set).
