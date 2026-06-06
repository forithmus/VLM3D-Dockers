# Forithmus-compatible evaluation container for VLM3D ct-volume-generation track.
# CUDA-base variant — the original python:3.11.5 base lacked libcudart.so so
# torch.cuda.is_available() returned False at runtime even on a GPU host.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        default-jre-headless \
        ffmpeg libsm6 libxext6 libxcb1 libgl1 libglib2.0-0 \
        ca-certificates git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/app/models /input/predictions /input/ground_truth /output && \
    chown -R user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# cu121-matched torch FIRST (so subsequent CT-CLIP/FVD installs don't pull a
# CPU-only or mismatched build).
RUN pip install --user --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Scoring code (flat repo layout — no src/ subdir)
COPY --chown=user:user \
    evaluation.py evaluate_fvd.py evaluate_clip.py evaluate_cpu.py evaluate_fid_2p5d_ct.py \
    metrics.py metrics3d.py compute_fid_2-5d_ct.py \
    /opt/app/

# Eval assets (flat repo layout — participants/maintainers populate these dirs
# locally per README before `docker build .`).
COPY --chown=user:user CT-CLIP/                        /opt/app/CT-CLIP/
COPY --chown=user:user FVD/                            /opt/app/FVD/
COPY --chown=user:user BiomedVLP-CXR-BERT-specialized/ /opt/app/models/BiomedVLP-CXR-BERT-specialized/
COPY --chown=user:user ctclip_weights/                 /opt/app/models/
COPY --chown=user:user ResNet50.pt                     /opt/app/
COPY --chown=user:user data_input.xlsx                 /opt/app/

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools && \
    pip install --user monai

WORKDIR /opt/app/CT-CLIP
RUN pip install --user -e ./transformer_maskgit && pip install --user -e ./CT_CLIP
WORKDIR /opt/app/FVD
RUN pip install --user -e .
WORKDIR /opt/app/
RUN pip install --user -r requirements.txt

# Pre-cache squeezenet1_1 (used by compute_fid_2-5d_ct.py as the
# feature extractor — radimagenet_resnet50 isn't usable because the upstream
# Google Drive link is rate-limited and produces an HTML page instead of the
# .pth file). evaluation.py is patched to pass --model_name squeezenet1_1.
RUN python -c "import torchvision; torchvision.models.squeezenet1_1(weights='DEFAULT'); print('squeezenet1_1 cached')"

ENTRYPOINT ["python", "-m", "evaluation"]
