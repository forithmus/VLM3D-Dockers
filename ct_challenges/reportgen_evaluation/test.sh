#!/usr/bin/env bash
# Local smoke test mirroring the Forithmus eval-container runtime:
#   /input/predictions  (RO)  ← test/predictions/
#   /input/ground_truth (RO)  ← test/ground_truth/
#   /output             (RW)  ← test_output/
# Network is disabled. No GPU is needed for RadBERT BERT-base on a few cases.

set -euo pipefail
HERE="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

rm -rf "$HERE/test_output"
mkdir -p "$HERE/test_output"
# Container runs as evaluator (uid 999); host's test_output must be writable.
# On Forithmus, /output is mounted with correct perms by the platform.
chmod 777 "$HERE/test_output"

# Use the bundled 2 000-case GT and the 1-case smoke submission.
docker run --rm \
    --platform linux/amd64 \
    --memory=8g --memory-swap=8g \
    --network=none \
    --cap-drop=ALL --security-opt=no-new-privileges \
    --shm-size=128m --pids-limit=256 \
    -v "$HERE/test/predictions":/input/predictions:ro \
    -v "$HERE/test/ground_truth":/input/ground_truth:ro \
    -v "$HERE/test_output":/output \
    vlm3d-reportgen-eval:1.0

echo
echo "==== /output/metrics.json ===="
python3 -m json.tool "$HERE/test_output/metrics.json"
