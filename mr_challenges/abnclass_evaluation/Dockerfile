# MR-RATE abnormality-classification evaluation container.
# Self-contained: scores participant predictions.json against ground_truth.csv
# (32 labels; macro + per-label precision/recall/F1/accuracy/AUROC).
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy pandas scikit-learn

COPY evaluation.py calc_scores.py json_to_csv.py labels.py /opt/app/

# Platform contract: /input/predictions/*.json (participant submission),
# /input/ground_truth/ground_truth.csv, writes /output/metrics.json
ENTRYPOINT ["python3", "/opt/app/evaluation.py"]
