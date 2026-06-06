#!/usr/bin/env bash
# Package the BTB3D reportgen image for upload via `forithmus submit`.
set -euo pipefail
./build.sh
docker save reportgen-btb3d | gzip -c > reportgen-btb3d.tar.gz
echo "Wrote reportgen-btb3d.tar.gz — submit with:"
echo "  forithmus submit reportgen-btb3d.tar.gz --phase <phase> --tier gpu-l4-xl --weights weights.zip"
