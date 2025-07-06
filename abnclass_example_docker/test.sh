#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"

docker volume create ctlipro-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed

docker run --rm\
        --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="bridge" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v ctlipro-output-$VOLUME_SUFFIX:/output/ \
        ctlipro



docker run --rm \
        -v ctlipro-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/results.json | python -m json.tool

mkdir -p ./mid_slices
docker run --rm \
  -v ctlipro-output-$VOLUME_SUFFIX:/out \
  -v "$(pwd)/mid_slices":/host \
  busybox cp -r /out/mid_slices /host/

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm ctlipro-output-$VOLUME_SUFFIX
