# MR-RATE MIL classification baseline (code-only image; weights arrive as a
# separate zip mounted at /weights by the platform).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    nibabel scipy einops timm peft "transformers==4.55.4" pyyaml pandas

# vjepa2 hub code pre-seeded so torch.hub never goes online.
RUN mkdir -p /opt/torchhub/hub \
 && git clone --depth 1 https://github.com/facebookresearch/vjepa2 \
      /opt/torchhub/hub/facebookresearch_vjepa2_main \
 && rm -rf /opt/torchhub/hub/facebookresearch_vjepa2_main/.git
ENV TORCH_HOME=/opt/torchhub

COPY MR-RATE/contrastive-pretraining /opt/MR-RATE/contrastive-pretraining
COPY MR-RATE/report-generation-training /opt/MR-RATE/report-generation-training
COPY run_inference.py run_inference_common.py run_generation.py entrypoint.sh entrypoint_rg.sh /opt/app/

RUN groupadd -r worker && useradd -m -r -g worker worker \
 && mkdir -p /input /output /weights \
 && chown -R worker:worker /opt/app /input /output /weights
USER worker
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/opt/app/entrypoint_rg.sh"]
