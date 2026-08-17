# MR Volume Generation — Evaluation Container

The production scoring container for `mr-volume-generation`.

Contract (set up by the platform):

```
/input/predictions/    participant-generated *.nii.gz volumes
/input/ground_truth/   reference MR volumes (flat directory)
/output/metrics.json   {"metrics": {...}}
```

Each generated volume must carry the **same filename as its target**
(`{study_uid}_{modality}-raw-{plane}.nii.gz`) — the modality is parsed from
the name. Scoring streams one (real, generated) pair at a time: basic
metrics (MSE / PSNR / SSIM) plus a 2.5D FID over squeezenet1_1 features
(weights baked into the image at build time; execution runs offline).
