# MR Abnormality Classification — Example Algorithm Container

Reference baseline for the `mr-abnormality-classification` track: a V-JEPA2
video encoder with a frozen MIL head over each study's atlas-space series
(MR-RATE pipeline). Reads brain MR studies from `/input`, writes 32-label
probabilities to `/output/predictions.json`.

## Input contract (important)

`/input` holds **one un-extracted zip per study** on a cloud-storage FUSE
mount: `<STUDY>.zip` containing `<STUDY>/{img,atlas,seg}/*.nii.gz`
(native series, MNI152-1mm-registered series, brain/defacing masks).
`run_inference_common.py` shows the recommended pattern: **lazy per-study
extraction with background prefetch** — the extracted dataset (~350 GB) does
not fit the 200 GB job disk, so never extract everything up front.

## Output

Grand-Challenge-wrapped JSON:

```json
[{"outputs": [{"value": {"predictions": [
  {"input_image_name": "<STUDY>.nii.gz",
   "probabilities": {"acute_ischemic_stroke": 0.03, "...": 0.0}}
]}}]}]
```

One entry per study, a probability in [0,1] for each of the 32 labels.

## Build & submit

```bash
docker build -t mr-abnclass-baseline .
forithmus init mr-abnormality-classification
forithmus test mr-abnclass-baseline --timeout 1200
docker save mr-abnclass-baseline | gzip > image.tar.gz
forithmus submit image.tar.gz --tier gpu-a100-80 --time-budget 240 --weights weights.zip
```

Weights ship as a separate `weights.zip`, mounted extracted at
`$FORITHMUS_WEIGHTS_DIR` (see `entrypoint.sh` for the expected layout).
Contact the organizers via the challenge page for the baseline weight bundle.

Throughput on `gpu-a100-80`: ~10 s/study (~5.5 h for the full 2,029-study set).
