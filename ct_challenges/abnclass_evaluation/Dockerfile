# Forithmus-compatible evaluation container for VLM3D abnormality-classification track.
#
# Reads:
#   /input/predictions/<*.json>   (participant submission, mounted by platform)
#   /input/ground_truth/          (host GT, mounted by platform, NOT baked-in)
# Writes:
#   /output/metrics.json          (consumed by the leaderboard)
#
# Baked into the image:
#   /opt/app/<code>.py            (scoring scripts only — no model weights, no GT)

FROM python:3.10-slim

# Java for any nltk-style downstream metrics if they ever get added; kept
# minimal otherwise. Same convention as the reportgen eval image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless \
 && rm -rf /var/lib/apt/lists/*

# Non-root user (Forithmus requirement: USER instruction).
RUN groupadd -r evaluator \
 && useradd -m -r -g evaluator evaluator \
 && mkdir -p /opt/app /input/predictions /input/ground_truth /output \
 && chown -R evaluator:evaluator /opt/app /input /output

USER evaluator
WORKDIR /opt/app
ENV PATH="/home/evaluator/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --chown=evaluator:evaluator requirements.txt ./
RUN pip install --user --no-cache-dir --upgrade pip \
 && pip install --user --no-cache-dir -r requirements.txt

# Scoring code (flat in this directory).
COPY --chown=evaluator:evaluator evaluation.py calc_scores.py crg_score.py json_to_csv.py /opt/app/

ENTRYPOINT ["python", "-m", "evaluation"]
