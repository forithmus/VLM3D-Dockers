#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

MEM_LIMIT="8g"

docker volume create abnclass-output-$VOLUME_SUFFIX
# Do not change any of the parameters to docker run, these are fixed

docker run --rm \
        --gpus '"device=2"' \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v abnclass-output-$VOLUME_SUFFIX:/output/ \
        abnclass

docker run --rm \
        -v abnclass-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/metrics.json | python -m json.tool
        
docker volume rm abnclass-output-$VOLUME_SUFFIX
