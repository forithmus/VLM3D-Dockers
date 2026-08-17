# MR report-generation evaluation container.
# vLLM base carries torch+cuda; the Gemma extraction model itself is NOT baked
# in — it is staged under the ground-truth prefix (mounted read-only at
# /input/ground_truth/_assets/gemma) because eval containers get no /weights.
FROM vllm/vllm-openai:cu129-nightly-x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pycocoevalcap pandas scikit-learn

COPY src/ /opt/app/
RUN mkdir -p /input/predictions /input/ground_truth /output

# Vertex mounts the NVIDIA driver at /usr/local/nvidia (GKE convention); the
# vLLM image does not include it in the loader path, so torch sees no GPU.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ENTRYPOINT ["python3", "/opt/app/evaluation.py"]
