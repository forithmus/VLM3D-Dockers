# MR volume-generation baseline (MAISI diffusion + CXR-BERT text conditioning).
# Weights arrive as a separate zip via `forithmus submit --weights`.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir monai nibabel transformers einops scipy

# Offline at runtime: bake the CXR-BERT text encoder into the image at build time
ENV HF_HOME=/app/hf_cache
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True); \
  AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True)"

COPY NV-Generate-CTMR /app/NV-Generate-CTMR
COPY inference.py /app/inference.py
WORKDIR /app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/app/inference.py"]
