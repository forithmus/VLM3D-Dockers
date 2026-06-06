# VLM3D Challenge — Docker Examples & Submission Guide

This repository is the canonical reference for participating in the **VLM3D** (Vision-Language Models for 3D Medical Imaging) Challenge hosted on the [Forithmus Research Hub](https://research.forithmus.com). It ships reference algorithm containers, the production evaluation containers, sample inputs/ground-truth data, and the end-to-end recipe a participant uses to go from `git clone` to a score on the leaderboard.

---

## 1. Overview

VLM3D is a multi-track benchmark on 3D chest CT. Each track gives you the same kind of input (3D volumes or text prompts) and expects you to ship a self-contained Docker container that runs offline on the platform's GPUs.

Active tracks:

| Track       | Slug        | Task                                                  |
| ----------- | ----------- | ----------------------------------------------------- |
| abnclass    | `abnclass`  | 18-class multi-label chest CT abnormality classification |
| reportgen   | `reportgen` | Free-text radiology report generation from a CT volume |
| ctgen       | `ctgen`     | Text-to-3D-CT volume generation                       |

Deprecated (kept under `deprecated_challenges/` for reference only — not accepted by the platform):

* `abnloc` — CT abnormality localization.

---

## 2. Repository layout

```text
VLM3D-Dockers/
├── ct_challenges/                 # active tracks
│   ├── abnclass_evaluation/       # 18-class classification — eval container
│   ├── abnclass_example_docker/   # 18-class classification — baseline algorithm
│   ├── ctgen_evaluation/          # text-to-CT generation — eval container
│   ├── ctgen_example_docker/      # text-to-CT generation — baseline algorithm
│   ├── reportgen_evaluation/      # CT-to-report generation — eval container
│   └── reportgen_example_docker/  # CT-to-report generation — baseline algorithm
├── deprecated_challenges/         # NO LONGER ACCEPTED
│   ├── abnloc_evaluation/
│   ├── abnloc_example_docker/
│   └── abnloc_example_gt/
├── example_gt_data/               # sample inputs / GT for local testing
│   ├── classification_example/
│   ├── ct_generation_example/
│   └── report_generation_example/
└── README.md                      # this file
```

Each `*_example_docker/` directory is a runnable baseline — `Dockerfile`, `process.py`, `requirements.txt`, `build.sh`, `test.sh`, `export.sh`, and a `test/` folder with a dummy volume. Each `*_evaluation/` directory is the production scoring container that the platform invokes on your predictions.

---

## 3. Per-track summary

### 3.1 abnclass — CT Abnormality Classification

* **Input**: a directory of CT volumes at `/input/` as `.nii.gz` (the platform's format).
* **Output**: a single `/output/results.json` with one entry per volume, each containing a `probabilities` object covering all 18 pathology labels (Medical material, Arterial wall calcification, Cardiomegaly, Pericardial effusion, Coronary artery wall calcification, Hiatal hernia, Lymphadenopathy, Emphysema, Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Pleural effusion, Mosaic attenuation pattern, Peribronchial thickening, Consolidation, Bronchiectasis, Interlobular septal thickening).
* **Ranking metric**: `crg.CRG` (clinically-weighted relevance), reported alongside macro AUROC and macro F1.
* **Baseline shipped**: `ct_challenges/abnclass_example_docker/` — CT-CLIP CTViT visual encoder + Biomed-CXR-BERT text encoder + linear classifier head (`CT_LiPro_v2` checkpoint).

### 3.2 reportgen — CT Report Generation

* **Input**: a directory of CT volumes at `/input/` (`.nii.gz` from the platform, recursively discovered). Each filename's bare series UUID (suffix stripped) is used as `input_image_name`.
* **Output**: `/output/predictions.json` in the Grand-Challenge-wrapped schema the eval container consumes: `[{"outputs":[{"value":{"name":"Generated reports","type":"Report generation","version":"1.0","generated_reports":[...]}}]}]`.
* **Ranking metric**: `crg.CRG` (clinically-weighted relevance), with `generation` (BLEU / ROUGE-L / METEOR / CIDEr via pycocoevalcap) and `classification.macro` (RadBERT-based label inference: F1, AUROC, recall, accuracy, precision) reported alongside.
* **Baseline shipped**: `ct_challenges/reportgen_example_docker/` — **BTB3D**, the deployed thin baseline that scored **CRG ≈ 0.346** in production: a MAGViT-2 visual tokenizer + LoRA-fine-tuned Llama-3.1-8B. Thin image (weights mounted from `/weights`). `predict.py` checkpoints completed volumes to `/checkpoint` for SIGTERM-resumability but runs inference fail-loud (no try/except around the per-volume loop).

### 3.3 ctgen — CT Volume Generation

* **Input**: a single text file at `/input/prompts.json`, a JSON array of `{"input_image_name": "<basename>", "report": "<radiology text>"}` objects.
* **Output**: loose `.nii.gz` volumes written to `/output/`, one per prompt, named `<input_image_name>.nii.gz`. The evaluation reads them directly (no archiving).
* **Ranking metric**: `metrics.FVD_CTNet` (Fréchet Video Distance over a CT-Net 3D backbone), with `CLIPScore` / `CLIPScore_I2I` / `CLIPScore_mean` and `FID_2p5D_{XY,XZ,YZ,Avg}` reported alongside.
* **Baseline shipped**: `ct_challenges/ctgen_example_docker/` — GenerateCT cascade (MaskGIT low-res transformer + diffusion super-resolution UNet) on a `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` base, with T5 and VGG16 pre-cached at build time.

---

## 4. Install the CLI

Everything below assumes the **Forithmus CLI** (`forithmus`). It is published on PyPI as `forithmus`.

```bash
pip install --upgrade forithmus
forithmus version          # prints the installed CLI version
```

Login is a browser-based OAuth flow against `https://research.forithmus.com/api`:

```bash
forithmus login            # opens a browser; paste the redirect token back
forithmus whoami           # confirms user + API endpoint
```

Credentials are persisted to `~/.forithmus/credentials.json`; the per-challenge context written by `forithmus init` lives at `~/.forithmus/config.json`. Sessions expire server-side — when commands start returning a 401-style "not authenticated" error, re-run `forithmus login`. `forithmus logout` clears the auth token (but keeps the `init` config).

---

## 5. End-to-end workflow

The full participant flow, from a fresh clone to a scored submission, using `abnclass` as the worked example. Substitute the appropriate paths for `reportgen` or `ctgen`.

The three challenge slugs are `ct-abnormality-classification`, `ct-report-generation`, and `ct-volume-generation`. Install the CLI first: `pip install --upgrade forithmus` (needs **≥ 0.1.10**).

```bash
# 5.1  Auth + pick the challenge
forithmus login
forithmus challenges
forithmus init ct-abnormality-classification   # auto-targets the current-edition phase (main-2026)
forithmus phases ct-abnormality-classification # the * row is what CLI commands target
forithmus status                               # confirm data + GT + eval are uploaded

# 5.2  Generate dummy inputs (+ expected output) that match the phase schema
forithmus generate               # writes .forithmus/test_data/{input,expected_output}

# 5.3  Build the baseline image
cd ct_challenges/abnclass_example_docker/
docker build -t abnclass-thin:latest .

# 5.4  Run it locally against the generated input + validate the output schema
forithmus test abnclass-thin:latest --timeout 600
# (or validate a directory of already-produced outputs:)
# forithmus validate ./output --input ./input

# 5.5  Package weights.zip — files must be at the ZIP ROOT, no parent dir
cd /path/to/raw_weights/
zip -r0 ../weights.zip ./*
unzip -l ../weights.zip          # sanity check: no leading raw_weights/ prefix

# 5.6  Save the Docker image as a .tar.gz the platform can ingest
cd /path/to/your/build/
docker save abnclass-thin:latest | gzip > submission.tar.gz

# 5.7  Submit
#  IMPORTANT: the -d / --desc string is your ALGORITHM NAME. It is shown
#  on the public leaderboard underneath your participant name, so give each
#  submission a clear, concise description of the method (e.g. the model
#  name + key variant). Keep it short — long descriptions are truncated in
#  the leaderboard UI.
forithmus submit submission.tar.gz \
    --phase main-2026 \
    --tier gpu-l4-xl \
    --time-budget 240 \
    --weights /path/to/weights.zip \
    -d "CT-CLIP classifier v1"

# 5.8  Watch it score
forithmus status
# (the leaderboard updates automatically once eval writes metrics.json)
```

For `reportgen` and `ctgen`, the only things that change are the challenge slug, the build directory (`ct_challenges/reportgen_example_docker/` or `ct_challenges/ctgen_example_docker/`), and the contents of `weights.zip`. `--phase` is optional once you've `init`'d — the CLI remembers the current-edition phase.

### What `forithmus generate` actually does

`generate` reads the phase's **data schema** (the platform infers it from the real test data + a baseline submission's output) and synthesizes a **fully fake** local test set under `.forithmus/test_data/`:

* `input/` — what your container reads at `/input`:
  * abnclass / reportgen → random-noise `.nii.gz` volumes with realistic headers (spacing/affine from the schema), named `case_001.nii.gz …`.
  * ctgen → a `prompts.json` with `case_001 … case_005` entries.
* `expected_output/` — the **shape** your container must produce at `/output`, with one entry per case:
  * abnclass → a single `results.json` with `predictions: [{input_image_name, probabilities: { …18 labels… }}]`.
  * reportgen → a single Grand-Challenge-wrapped `predictions.json` with `generated_reports: [{input_image_name, report}]`.

**The values are random** (probabilities are `rand()`, reports are placeholder strings) — they exist only to show you the exact JSON structure and field names your output must match. No real scans, reports, labels, or ground truth are ever materialized. `forithmus test` then runs your image against this fake `input/` and checks your `/output` against the `expected_output/` schema.

---

## 6. How the platform mounts your container at runtime

At dispatch time the platform appends a small POSIX-shell **trampoline** (`forithmus-entry.sh`) to your image via `crane append` + `crane mutate`. The mutation rewrites the image config so `ENTRYPOINT=/forithmus-entry.sh` and `USER=root`, and stashes your original `ENTRYPOINT+CMD` as `FORITHMUS_ORIG_CMDLINE`. Your image is otherwise untouched — the trampoline sets up the FUSE-backed mounts, fixes a few PYTHONPATH gotchas for user-site packages, then `eval exec`s your original command so your process runs as PID 1 (which is what lets Vertex flush gcsfuse writes cleanly on shutdown).

### Algorithm container (submission mode)

```text
/input/         read-only   FUSE → gs://forithmus-production-data/{challenge_id}/{phase_id}/
                            .nii.gz volumes (abnclass, reportgen)
                            or prompts.json (ctgen)
/output/        writable    FUSE → gs://forithmus-production-io/{challenge_id}/{submission_id}/predictions/
                            you MUST write results.json / *.nii.gz here
/weights/       read-only   from gs://forithmus-production-io/{challenge_id}/weights/{submission_id}/weights.zip
                            extracted to local disk, symlinked at /weights
/checkpoint/    writable    optional, for spot-retry resumability
/tmp/           writable    1 GB tmpfs (dev) / 200 GB pd-ssd (prod) — use for scratch
```

The trampoline (`forithmus-entry.sh`) exports the contract env vars that point at the resolved runtime paths: `FORITHMUS_INPUT`, `FORITHMUS_OUTPUT`, `FORITHMUS_WEIGHTS`, `FORITHMUS_CHECKPOINT`. (The Vertex job dispatcher additionally injects run-scoped vars such as the attempt index and remaining-minutes budget into the container environment.)

### Evaluation container (eval mode, host-only)

```text
/input/predictions/    read-only   FUSE → your submission's predictions/ prefix
/input/ground_truth/   read-only   FUSE → the hidden per-phase GT bucket (eval-SA only)
/output/               writable    LOCAL disk; the trampoline uploads files to
                                   gs://.../evaluation/ via the GCS JSON API after the eval
                                   process exits (this is NOT a FUSE mount — by design,
                                   because FUSE writes don't flush before Vertex teardown)
```

`/output` in the eval container is **local disk** specifically because a proven race made FUSE writes vanish during Vertex container teardown. The trampoline walks `/tmp/forithmus/output` after eval exits, POSTs each file via `storage.googleapis.com/upload/storage/v1/b/<bucket>/o`, and forces a non-zero exit code (rc=75) if any upload fails — so the leaderboard never sees a half-written `metrics.json`.

---

## 7. Packaging `weights.zip`

Build `weights.zip` with your model files **at the ZIP root, with no parent directory**. The trampoline copies the ZIP off FUSE, unzips it into `/tmp/forithmus/weights`, and symlinks `/weights` at the extracted tree. So if your zip contains `CT_LiPro_v2.pt`, it shows up at runtime as `/weights/CT_LiPro_v2.pt`.

```bash
cd /path/to/raw_weights/        # the DIRECTORY whose CONTENTS are your model files
zip -r0 ../weights.zip ./*      # store-only (-0), files at root
unzip -l ../weights.zip         # must show no leading "raw_weights/" prefix
```

### abnclass example layout

`weights.zip` must contain at its top level:

```text
BiomedVLP-CXR-BERT-specialized/    # HF model directory (tokenizer + BertModel)
CT_LiPro_v2.pt                     # ImageLatentsClassifier state-dict
```

At runtime the thin algorithm's `entrypoint.sh` symlinks each into `/opt/app/models/` so the unchanged `process.py` keeps loading from its hardcoded paths:

```bash
ln -sf /weights/BiomedVLP-CXR-BERT-specialized /opt/app/models/BiomedVLP-CXR-BERT-specialized
ln -sf /weights/CT_LiPro_v2.pt                 /opt/app/models/CT_LiPro_v2.pt
exec python /opt/app/process.py
```

### ctgen example layout

```text
ctvit_pretrained.pt
transformer_pretrained.pt
superres_pretrained.pt
```

The `ctgen_example_docker/entrypoint.sh` symlinks each into `/opt/app/models/` and then execs `python /opt/app/process.py`.

---

## 8. The thin-image pattern (recommended)

The platform's image-size validator rejects any submission tarball larger than **15 GB compressed** (`PIPELINE_MAX_CONTAINER_SIZE_GB`). Model weights for a CT-CLIP-class encoder + LLM head easily blow past this. The pattern that works:

* **Thin image** ships your code, dependencies, and pre-cached non-weight assets (T5, VGG16, RadBERT tokenizer JSONs, etc.) — no model checkpoints.
* **`weights.zip`** ships separately via `--weights` (cap is 100 GB) and is mounted at `/weights` by the trampoline.
* **`entrypoint.sh`** runs at container start, symlinks the expected files from `/weights` into your hardcoded `/opt/app/models/` paths, then `exec`s your original `process.py`.

Concretely, your `Dockerfile` simply drops the `COPY models/ /opt/app/models/` line and adds:

```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER appuser                     # validator rejects effective USER=root
ENTRYPOINT ["/entrypoint.sh"]
```

Copyable `entrypoint.sh` stub (POSIX `/bin/sh`, no bash-isms):

```sh
#!/bin/sh
set -eu

MODELS_DIR=/opt/app/models
mkdir -p "$MODELS_DIR"

# Symlink everything in /weights into /opt/app/models so the unmodified
# process.py keeps loading from its hardcoded paths.
for f in /weights/*; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    ln -sf "$f" "$MODELS_DIR/$name"
done

# Defensive: some user Dockerfiles aggressively clean /; ensure /output exists.
mkdir -p /output

exec python /opt/app/process.py
```

---

## 9. Local-testing checklist

Before you `forithmus submit`, verify your container:

* **Runs as a non-root user.** The platform validator REJECTS images whose effective `USER` is root (0). Declare `USER appuser` (or any UID > 0) in your Dockerfile. The trampoline will mutate the runtime user to root for FUSE writes — but it does that AFTER validation passes, so your image must still be configured as non-root.
* **Exits non-zero on any failure.** No `try/except` around the inference loop "to skip bad volumes" — that hides OOMs, missing weights, and tensor-shape bugs. The platform's failure path expects a non-zero exit code.
* **Works offline.** There is no network egress at runtime — the Vertex job is on `GCP_PIPELINE_NETWORK`, an isolated VPC with no external IP and Private Google Access only. Pre-cache HuggingFace models, NLTK data, Torch Hub weights, fonts, etc. at build time and set `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` in the image.
* **Reads `.nii.gz` from `/input`.** The platform ships volumes as `.nii.gz`; the shipped examples read `.nii.gz` only — match that exactly so there is no ambiguity.
* **Writes to `/output` per the track's schema.** Exactly:
  * abnclass → `/output/results.json` with all 18 labels per case.
  * reportgen → `/output/predictions.json` in the Grand-Challenge-wrapped schema (see §3.2).
  * ctgen → loose `/output/<input_image_name>.nii.gz` volumes, one per prompt.
  Always call `os.makedirs("/output", exist_ok=True)` first — the trampoline guarantees the SYMLINK, not the leaf directory.
* **Uses a CUDA-capable base.** `python:3.x` slim images lack `libcudart` and silently flip `torch.cuda.is_available()` to `False`. Use `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`.

`forithmus test <image>` does most of these checks for you against the dummy data from `forithmus generate`; `forithmus validate <output_dir>` will re-check the output schema without re-running Docker.

---

## 10. Compute tiers

Tiers are billed per minute on the slug you pick with `--tier`. Total cost = (validate + run + eval) minutes × tier $/hr. Pull the live table any time with `forithmus tiers`.

| Slug          | Hardware                              | Host RAM | $/hr   |
| ------------- | ------------------------------------- | -------- | ------ |
| `cpu-4`       | 4 vCPU                                | 16 GB    | $0.23  |
| `cpu-8`       | 8 vCPU                                | 32 GB    | $0.46  |
| `cpu-16`      | 16 vCPU                               | 64 GB    | $0.91  |
| `cpu-32`      | 32 vCPU                               | 128 GB   | $1.82  |
| `gpu-t4`      | NVIDIA T4 (16 GB VRAM)                | 16 GB    | $0.65  |
| `gpu-l4`      | NVIDIA L4 (24 GB VRAM)                | 16 GB    | $0.85  |
| **`gpu-l4-xl`** | **NVIDIA L4 (24 GB VRAM)**         | **32 GB** | **$1.30** |
| `gpu-v100`    | NVIDIA V100 (32 GB VRAM)              | —        | $3.43  |
| `gpu-a100-40` | NVIDIA A100 40GB                      | —        | $4.41  |
| `gpu-a100-80` | NVIDIA A100 80GB                      | —        | $6.03  |
| `gpu-h100`    | NVIDIA H100 80GB                      | —        | $13.00 |
| `gpu-2xa100`  | 2× A100 40GB (80 GB VRAM)             | —        | $8.82  |
| `gpu-4xa100`  | 4× A100 40GB (160 GB VRAM)            | —        | $17.63 |
| `gpu-8xa100`  | 8× A100 40GB (320 GB VRAM)            | —        | $35.26 |
| `gpu-8xh100`  | 8× H100 80GB (640 GB VRAM)            | —        | $105.40|
| `tpu-v5e-1`   | 1× TPU v5e (16 GB HBM)                | —        | $1.56  |
| `tpu-v5e-4`   | 4× TPU v5e (64 GB HBM)                | —        | $6.24  |
| `tpu-v4-4`    | 4× TPU v4 (128 GB HBM)                | —        | $16.70 |
| `tpu-v5p-4`   | 4× TPU v5p (380 GB HBM)               | —        | $21.80 |
| `tpu-v6e-4`   | 4× TPU v6e Trillium (128 GB HBM)      | —        | $14.00 |

**Default suggestion**: `gpu-l4-xl` — same L4 GPU as `gpu-l4` but with 32 GB host RAM, which is the cheapest tier that comfortably loads a CT volume + a 24 GB VRAM model without OOMing the host on preprocessing. Bump to `gpu-a100-80` when you need an 8B-parameter LLM head or a multi-GPU training tier.

---

## 11. Submission lifecycle

A `forithmus submit` walks through these stages — the CLI streams transitions back as the backend advances them:

1. **Upload** — chunked resumable upload of `submission.tar.gz` (and `weights.zip` if `--weights`) to a signed-URL GCS staging bucket. `--reuse-weights <prior_id>` skips the weights re-upload by re-referencing the prior submission's bundle.
2. **Build** — Cloud Build loads the tarball, has `crane` append the trampoline layer and rewrite the entrypoint, and pushes the result to per-challenge Artifact Registry.
3. **Validate** — runs the built image with a sentinel probe to confirm it starts, declares a version banner, and exits cleanly. (You pay validate-time too.)
4. **ClamAV scan** — both the uploaded tarball and the built image layers are scanned for malware before any GPU is touched. The scanner's timeout was raised to 50 min — if it returns "Scanner unreachable", just retry.
5. **Run** — Vertex AI custom job on the requested `--tier`, with the FUSE mounts described in §6. The watcher polls Vertex `JOB_STATE` every minute and streams it back to the CLI. Capacity-constrained tiers (a2/a3, A100 80GB, H100) may sit in `PENDING` while Vertex waits for the accelerator — this is normal. Spot preemptions are auto-retried with checkpoint restore; after N preemptions the orchestrator falls back to on-demand, and your `--time-budget` is preserved across retries.
6. **Output upload** — the trampoline's `exec` keeps your process as PID 1 long enough for Vertex to flush `/output` via gcsfuse to `gs://.../{submission_id}/predictions/`.
7. **Eval** — the host's eval container (uploaded via `forithmus upload-eval`) runs against `/input/predictions/` and the hidden `/input/ground_truth/`. It writes `/output/metrics.json` to local disk, which the trampoline then POSTs to `gs://.../{submission_id}/evaluation/` via the GCS JSON API (see §6).
8. **Score** — backend parses `metrics.json`, validates finiteness (NaN/Inf in the primary score forces FAILED), persists the metrics row, and updates the leaderboard. Your submission appears on the leaderboard with your **`-d` / `--desc` string shown as the algorithm name** beneath your participant name — so always pass a clear, concise `-d` describing the method.

Phase-level timeouts and the per-tier wall-clock budget (`--time-budget`) bound the run-phase duration. The eval container has its own 2 GB output cap and its `metrics.json` is parsed for finiteness before it's persisted.

---

## 12. Re-using weights from a previous submission

Re-uploading a multi-GB `weights.zip` for every code-only iteration is wasteful. If your weights haven't changed, pass `--reuse-weights <prior_submission_id>` instead of `--weights`:

```bash
forithmus submit submission_v2.tar.gz \
    --phase main-2026 \
    --tier gpu-l4-xl \
    --reuse-weights 0f1a2b3c-4d5e-6f70-8a90-1b2c3d4e5f60 \
    -d "abnclass thin v2 — code-only fix, same weights as v1"
```

`--weights` and `--reuse-weights` are mutually exclusive.

**Warning**: only reuse weights from a submission that **scored successfully**. Failed submissions get their prediction outputs cleaned up by the orchestrator after the retention window; the weights bundle itself lives under `gs://forithmus-production-io/{challenge_id}/weights/{prior_id}/weights.zip`, but if that prior submission was failed and garbage-collected, the reuse pointer breaks at mount time and your new run will fail before it starts.

---

## 13. Troubleshooting

| Symptom (CLI / logs)                                                | Cause + fix                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Container validation failed: Scanner unreachable`                  | The ClamAV scanner timed out under load. Retry — the scanner timeout was raised to 50 min, and the build job is idempotent.                                                                                                                                              |
| `torch.cuda.is_available()` returns `False` inside the container    | Your base image lacks `libcudart`. Switch from `python:3.x` to `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (or another CUDA-runtime base) and reinstall your Python deps on top.                                                                                  |
| `Could not connect to huggingface.co` / `OSError: [Errno -3]`       | There is no DNS to non-Google endpoints at runtime. Pre-download every HF model at build time (`huggingface-cli download …`), bake them into the image, and set `ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`.                                                              |
| `Validator rejected image: effective USER is root (0)`              | Add a non-root `USER` directive in your Dockerfile (e.g. `RUN useradd -m appuser && USER appuser`). The trampoline mutates the runtime user back to root for FUSE writes — but the validator runs BEFORE the mutation and refuses root-as-shipped.                          |
| `Image too large (>15 GB)`                                          | Move model weights out of the image and into `weights.zip`; submit with `--weights weights.zip`. See §7 + §8 for the thin-image pattern.                                                                                                                                   |
| `CUDA out of memory` partway through                                | Don't wrap the inference loop in `try/except` to "skip" the failing case — let the error propagate so the platform fails fast and the run is visible. Then pick a larger tier (`gpu-l4-xl` → `gpu-a100-80`) or stream the volume through your model in tiles.               |
| `/input/predictions` is empty during eval                           | Your algorithm container exited cleanly but didn't write anything. Confirm you wrote to `/output/<expected_name>` (not `/predictions/` or `/tmp/`). The trampoline `link_force`s the FUSE symlink — a baked `mkdir /input/predictions` in your Dockerfile won't shadow it.  |
| Submission stuck in `PENDING` on a heavy GPU tier                   | Vertex is queuing for accelerator capacity. Either wait, or resubmit on a less-constrained tier (`gpu-l4-xl`, `gpu-a100-40`).                                                                                                                                              |
| `import pandas` (or any user-site package) fails at runtime         | The trampoline globs `/home/*/.local/lib/python*/site-packages` and prepends them to `PYTHONPATH` automatically. If your image installs packages somewhere weirder (e.g. `/srv/.venv/`), add it to `ENV PYTHONPATH=…` in your Dockerfile.                                   |
| Eval `metrics.json` shows your primary score as `null`              | The primary score was NaN/Inf and the backend nulled it out; this also forces the submission to FAILED. Check that your predictions cover all GT cases and don't divide by zero on degenerate inputs.                                                                      |

---

## 14. Getting help

* **Platform issues** (auth, builds, scanner timeouts, leaderboard) — [https://research.forithmus.com](https://research.forithmus.com) → the support widget, or the in-platform Discord linked from the challenge page.
* **Challenge-specific questions** (label semantics, GT format, baseline reproducibility) — the per-track README inside `ct_challenges/<track>_example_docker/` and `ct_challenges/<track>_evaluation/`, then the challenge's discussion forum.
* **CLI bugs** — run `forithmus version`, attach the full stderr, and file under the Forithmus Research Hub support channel.

Good luck — and may your dice scores be high.
