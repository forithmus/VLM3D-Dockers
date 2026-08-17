#!/bin/bash
# Report-generation baseline: same config retargeting, different entry script.
set -e
W="${FORITHMUS_WEIGHTS_DIR:-/weights}"
SRC="$W/bundle/configs/encoder_late_attn_cross_attn_minmax_pt2.yaml"
CFG="/tmp/encoder.local.yaml"
sed -e "s|__BUNDLE__|$W/bundle|g" \
    -e "s|__UPSTREAM_ROOT__|/opt/MR-RATE/contrastive-pretraining|g" \
    -e "s|__LLM_PATH__|$W/medgemma-4b-it|g" \
    -e "s|__VJEPA_BACKBONE__|$W/vjepa2_1_vitg_384.pt|g" \
    -e "s|__MRI_ROOT__|/input|g" \
    -e "s|__OUTPUT_DIR__|/output|g" \
    "$SRC" > "$CFG"
export ENCODER_CONFIG="$CFG"
exec python /opt/app/run_generation.py
