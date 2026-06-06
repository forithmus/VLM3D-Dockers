#!/usr/bin/env bash
# Thin-abnclass entrypoint:
#   1. bridge /weights -> /opt/app/models so process.py runs unchanged
#   2. run process.py (writes /output/results.json — flat schema)
#   3. wrap the result in the Grand-Challenge format the platform's eval
#      container expects: [{"outputs":[{"value":<inner>}]}] at
#      /output/predictions.json — same shape that vlm3d-reportgen-eval
#      consumed for the BTB3D scored runs.
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"
MODELS_DIR="/opt/app/models"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"

# Required by process.py
#   TEXT_MODEL_PATH = /opt/app/models/BiomedVLP-CXR-BERT-specialized
#   CLASSIFIER_PATH = /opt/app/models/CT_LiPro_v2.pt
for name in BiomedVLP-CXR-BERT-specialized CT_LiPro_v2.pt; do
    src="$WEIGHTS_DIR/$name"
    dst="$MODELS_DIR/$name"
    if [ -e "$src" ]; then
        ln -sfn "$src" "$dst"
    else
        echo "[entrypoint] WARNING: $src not present in /weights mount" >&2
    fi
done

echo "[entrypoint] abnclass thin container starting"
echo "[entrypoint]   /weights contents:"
ls -la "$WEIGHTS_DIR" 2>&1 | head -10 | sed 's/^/[entrypoint]     /' || true
echo "[entrypoint]   /opt/app/models:"
ls -la "$MODELS_DIR" 2>&1 | head -10 | sed 's/^/[entrypoint]     /' || true

set +e
python /opt/app/process.py
RC=$?
set -e
echo "[entrypoint] process.py exit=$RC"

# Wrap into GC format and emit predictions.json (the canonical filename
# the platform's eval reader expects).
python <<'PY'
import json, os, pathlib, sys

out_dir = pathlib.Path(os.environ.get("OUTPUT_DIR", "/output"))
src = out_dir / "results.json"
dst = out_dir / "predictions.json"

if src.exists():
    try:
        inner = json.loads(src.read_text())
    except Exception as e:
        print(f"[entrypoint] FAILED to parse {src}: {e}", file=sys.stderr)
        sys.exit(2)
    wrapped = [{"outputs": [{"value": inner}]}]
    dst.write_text(json.dumps(wrapped))
    print(f"[entrypoint] wrapped {src} -> {dst} (GC format, {len(inner.get('predictions', []))} predictions)")
else:
    # process.py exited without writing — emit empty schema so the eval
    # container sees SOMETHING (it will compute zero on it but the run
    # won't crash trying to read a missing file).
    empty = [{"outputs": [{"value": {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"major": 1, "minor": 0},
        "predictions": [],
    }}]}]
    dst.write_text(json.dumps(empty))
    print(f"[entrypoint] results.json missing — wrote empty predictions.json", file=sys.stderr)
PY

exit $RC
