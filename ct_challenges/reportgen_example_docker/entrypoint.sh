#!/usr/bin/env bash
# Submission container entrypoint.
# - Forwards SIGTERM to predict.py so its handler can flush state.
# - On exit, makes sure /output/predictions.json exists (writes empty schema if not).

set -uo pipefail

INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/checkpoint}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

echo "[entrypoint] BTB3D submission container starting"
echo "[entrypoint]   INPUT_DIR     = $INPUT_DIR"
echo "[entrypoint]   OUTPUT_DIR    = $OUTPUT_DIR"
echo "[entrypoint]   CHECKPOINT_DIR= $CHECKPOINT_DIR"
echo "[entrypoint]   BTB3D_QUANT   = ${BTB3D_QUANT:-4bit}"

# Run predict.py in the foreground; trap SIGTERM and pass it through.
python /opt/btb3d/predict.py &
PID=$!

term_handler() {
    echo "[entrypoint] SIGTERM received, forwarding to predict.py (pid=$PID)"
    kill -SIGTERM "$PID" 2>/dev/null || true
    # Wait up to 25s for predict to flush; platform gives us 30s.
    for _ in $(seq 1 25); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
}
trap term_handler SIGTERM

wait "$PID"
RC=$?

# Defense: ensure predictions.json exists (empty if predict.py never wrote it).
# Match the GC-wrapped schema the eval container expects.
if [ ! -f "$OUTPUT_DIR/predictions.json" ]; then
    echo "[entrypoint] predictions.json missing — writing empty schema"
    cat > "$OUTPUT_DIR/predictions.json" <<EOF
[{"outputs": [{"value": {"name": "Generated reports", "type": "Report generation", "version": "1.0", "generated_reports": []}}]}]
EOF
fi

echo "[entrypoint] exit $RC"
exit "$RC"
