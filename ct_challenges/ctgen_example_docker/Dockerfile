#syntax=docker/dockerfile:1.6
# Thin variant of the GenerateCT ctgen algorithm. No baked model weights.
# Weights uploaded separately to Forithmus via --weights ctgen-weights.zip
# and mounted at /weights at runtime.
#
# Base image carries CUDA 12.1 runtime libs that torch 2.4.1+cu121 needs to
# talk to Vertex's GPU driver. The plain python: image does NOT — torch's
# torch.cuda.is_available() then returns False because libcudart.so et al.
# are missing on the system. Same pattern as btb3d-reportgen-thin.
#
# /weights expected layout:
#   /weights/ctvit_pretrained.pt
#   /weights/transformer_pretrained.pt
#   /weights/superres_pretrained.pt

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ca-certificates \
        ffmpeg libsm6 libxext6 libxcb1 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN useradd --create-home --uid 1000 user && \
    mkdir -p /opt/app/models /input /output && \
    chown -R user:user /opt/app /input /output

USER user
WORKDIR /opt/app/
ENV PATH="/home/user/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1

# GenerateCT source — small, no weights.
COPY --chown=user:user GenerateCT /opt/app/GenerateCT

# cu121-matched torch FIRST.
RUN pip install --user --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY --chown=user:user requirements.txt /opt/app/
RUN pip install --user --no-cache-dir -r requirements.txt

WORKDIR /opt/app/GenerateCT
RUN pip install --user --no-cache-dir -e ./transformer_maskgit && \
    pip install --user --no-cache-dir -e ./super_resolution
WORKDIR /opt/app/

# Pre-download every model the runtime would fetch from the network
# (Forithmus runs the algorithm with no egress).
#   - google/t5-v1_1-base (super_resolution.t5 import time)
#   - VGG16 (ctvit instantiates torchvision.models.vgg16(pretrained=True))
RUN pip install --user --no-cache-dir sentencepiece
RUN python -c "from transformers import T5Config, T5Tokenizer, T5EncoderModel; \
    m='google/t5-v1_1-base'; \
    T5Config.from_pretrained(m); \
    T5Tokenizer.from_pretrained(m); \
    T5EncoderModel.from_pretrained(m); \
    print('T5 cached')"
RUN python -c "import torchvision; torchvision.models.vgg16(weights='DEFAULT'); print('VGG16 cached')"

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
