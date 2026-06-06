#!/usr/bin/env bash
set -euo pipefail
HERE="$( cd "$(dirname "$0")" ; pwd -P )"
docker build --platform linux/amd64 -t vlm3d-reportgen-eval:1.0 "$HERE"
