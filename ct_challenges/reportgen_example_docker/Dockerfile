#syntax=docker/dockerfile:1.6
# Thin variant of submission/Dockerfile: no baked weights.
# Weights are uploaded separately to Forithmus via `forithmus submit --weights weights.zip`
# and mounted at /weights inside the container at runtime.
#
# Build:  bash scripts/build-thin-submission.sh
# Upload: WEIGHTS_ZIP=/path/to/weights.zip bash scripts/upload-submission.sh
#         (after the first upload, switch to REUSE_WEIGHTS=<prev-submission-id>)

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# -------- system deps (early, rarely changes) --------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /opt/btb3d

# -------- python deps (cu121-matched torch first, then the pinned set) --------
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# transformer_maskgit: see Dockerfile (non-thin) for the rationale on
# install-then-overwrite. Behaviour is identical here.
RUN pip install --no-cache-dir \
        'git+https://github.com/ibrahimethemhamamci/CT-CLIP.git#subdirectory=transformer_maskgit' && \
    rm -rf /root/.cache/pip
COPY src/transformer_maskgit/ /usr/local/lib/python3.10/dist-packages/transformer_maskgit/

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libxcb1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# -------- BTB3D source --------
COPY src/encoder-decoder/   ./encoder-decoder/
COPY src/report-generation/ ./report-generation/
RUN pip install --no-cache-dir -e ./report-generation/ && \
    rm -rf /root/.cache/pip

# -------- entrypoint --------
COPY predict.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Forithmus mounts the uploaded weights.zip contents at /weights.
# predict.py reads WEIGHTS_DIR; pointing it there is the only adapter needed.
#
# BTB3D_QUANT is a build arg so the quant default is a build flag, not a code
# edit. Default bf16 matches the bd5107a0 ("thin bf16 v13") submission; build
# with --build-arg BTB3D_QUANT=4bit for the T4-targeted 4-bit variant.
ARG BTB3D_QUANT=bf16
ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    CHECKPOINT_DIR=/checkpoint \
    WEIGHTS_DIR=/weights \
    BTB3D_QUANT=${BTB3D_QUANT} \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# Non-root user (Forithmus validator rejects root).
RUN useradd --create-home --uid 1000 runner && \
    chown -R runner:runner /opt/btb3d
USER runner

ENTRYPOINT ["/opt/btb3d/entrypoint.sh"]
