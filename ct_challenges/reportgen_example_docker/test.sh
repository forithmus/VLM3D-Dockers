#!/usr/bin/env bash
# Local smoke test for the BTB3D reportgen example.
# Mounts ./test as /input, runs inference, and pretty-prints the
# Grand-Challenge-wrapped /output/predictions.json the eval container expects.
#
# Weights are NOT baked into the thin image — mount them at /weights
# (see ../../README.md for the weights.zip packaging flow). For a quick
# build-only smoke test you can omit -v on /weights, but inference will
# fail loudly without the model weights present.
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
./build.sh

MEM_LIMIT="30g"
docker volume create reportgen-output

docker run --rm \
        --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --shm-size="128m" \
        -v "$SCRIPTPATH/test/:/input/" \
        -v "$SCRIPTPATH/weights/:/weights/" \
        -v reportgen-output:/output/ \
        reportgen-btb3d

docker run --rm -v reportgen-output:/output/ \
        python:3.10-slim cat /output/predictions.json | python -m json.tool

docker volume rm reportgen-output
