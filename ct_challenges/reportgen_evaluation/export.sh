#!/usr/bin/env bash
set -euo pipefail
HERE="$( cd "$(dirname "$0")" ; pwd -P )"
./build.sh
docker save vlm3d-reportgen-eval:1.0 | gzip -c > "$HERE/vlm3d-reportgen-eval-1.0.tar.gz"
ls -la "$HERE/vlm3d-reportgen-eval-1.0.tar.gz"
echo
echo "Upload to Forithmus via:"
echo "  forithmus upload-eval $HERE/vlm3d-reportgen-eval-1.0.tar.gz \\"
echo "      --challenge ct-report-generation --phase main-2026"
