#syntax=docker/dockerfile:1.6
# Thin variant of the CT-LiPro abnclass classifier. No baked model weights.
# Weights are uploaded separately to Forithmus via:
#   forithmus submit ... --weights abnclass-weights.zip
# (or --reuse-weights <prev-sub-id>) and mounted at /weights at runtime.
#
# Base image carries CUDA 12.1 runtime libs that torch 2.4.1+cu121 needs to
# talk to Vertex's L4 driver. The plain python: image does NOT — torch's
# torch.cuda.is_available() then returns False because libcudart.so et al.
# are missing on the system. Same pattern as btb3d-reportgen-thin.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps + Python 3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ca-certificates default-jre-headless \
        ffmpeg libsm6 libxext6 libxcb1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Non-root user (Forithmus validator rejects root).
RUN useradd --create-home --uid 1000 user && \
    mkdir -p /opt/app/models /input /output && \
    chown -R user:user /opt/app /input /output

USER user
WORKDIR /opt/app/
ENV PATH="/home/user/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1

# CT-CLIP source — small, no weights.
COPY --chown=user:user CT-CLIP /opt/app/CT-CLIP

# Pin cu121-matched torch first (matches Vertex L4 driver). Then the rest.
RUN pip install --user --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY --chown=user:user requirements.txt /opt/app/
# Plain pip (NOT piptools sync) so torch stays pinned.
RUN pip install --user --no-cache-dir -r requirements.txt

WORKDIR /opt/app/CT-CLIP
RUN pip install --user --no-cache-dir -e ./transformer_maskgit && \
    pip install --user --no-cache-dir -e ./CT_CLIP
WORKDIR /opt/app/

# Pre-download HF assets — Forithmus no-egress sandbox at runtime.
# ct_clip.py:585 hardcodes BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized')
RUN python -c "from transformers import BertTokenizer, BertModel; \
    m='microsoft/BiomedVLP-CXR-BERT-specialized'; \
    BertTokenizer.from_pretrained(m); \
    BertModel.from_pretrained(m); \
    print('BiomedVLP cached')"

ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user entrypoint.sh /opt/app/
USER root
RUN chmod +x /opt/app/entrypoint.sh
USER user

ENTRYPOINT ["/opt/app/entrypoint.sh"]
