#!/usr/bin/env bash
# Thin-ctgen entrypoint:
#   1. bridge /weights -> /opt/app/models so process.py runs unchanged
#   2. run process.py (writes loose .nii.gz volumes to /output/, one per prompt)
# No output wrapping needed: the platform shuttles /output/*.nii.gz to
# /input/predictions/ and the eval reads the .nii.gz volumes directly.
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"
MODELS_DIR="/opt/app/models"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"

# Required by process.py (hardcoded paths under /opt/app/models)
for name in ctvit_pretrained.pt transformer_pretrained.pt superres_pretrained.pt; do
    src="$WEIGHTS_DIR/$name"
    dst="$MODELS_DIR/$name"
    if [ -e "$src" ]; then
        ln -sfn "$src" "$dst"
    else
        echo "[entrypoint] WARNING: $src not present in /weights mount" >&2
    fi
done

echo "[entrypoint] ctgen thin container starting"
echo "[entrypoint]   /weights contents:"
ls -la "$WEIGHTS_DIR" 2>&1 | head -10 | sed 's/^/[entrypoint]     /' || true
echo "[entrypoint]   /opt/app/models:"
ls -la "$MODELS_DIR" 2>&1 | head -10 | sed 's/^/[entrypoint]     /' || true
echo "[entrypoint]   /input:"
ls -la /input 2>&1 | head -10 | sed 's/^/[entrypoint]     /' || true

exec python /opt/app/process.py
