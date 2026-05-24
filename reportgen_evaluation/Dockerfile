# Forithmus-compatible evaluation container for VLM3D report-generation track.
#
# Reads:
#   /input/predictions/<*.json>   (participant submission, mounted by platform)
#   /input/ground_truth/          (host GT, mounted by platform, NOT baked-in)
# Writes:
#   /output/metrics.json          (consumed by the leaderboard)
#
# Baked into the image:
#   /opt/app/<code>.py            (scoring scripts)
#   /opt/app/RadBertClassifier.pth (fine-tuned head, evaluation asset)
#   /opt/app/radbert_local/       (base RadBERT tokenizer + weights, offline)

FROM python:3.10-slim

# Java for METEOR (pycocoevalcap)
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless \
 && rm -rf /var/lib/apt/lists/*

# Non-root user (Forithmus requirement: USER instruction)
RUN groupadd -r evaluator \
 && useradd -m -r -g evaluator evaluator \
 && mkdir -p /opt/app /input/predictions /input/ground_truth /output \
 && chown -R evaluator:evaluator /opt/app /input /output

USER evaluator
WORKDIR /opt/app
ENV PATH="/home/evaluator/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY --chown=evaluator:evaluator requirements.txt ./
RUN pip install --user --no-cache-dir --upgrade pip \
 && pip install --user --no-cache-dir -r requirements.txt

# Scoring code
COPY --chown=evaluator:evaluator src/ /opt/app/

# Evaluation models (NOT ground truth)
COPY --chown=evaluator:evaluator assets/RadBertClassifier.pth /opt/app/
COPY --chown=evaluator:evaluator assets/radbert_local/        /opt/app/radbert_local/

ENTRYPOINT ["python", "-m", "evaluation"]
