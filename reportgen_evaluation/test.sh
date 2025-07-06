#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

MEM_LIMIT="8g"

docker volume create reportgen-output-$VOLUME_SUFFIX
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
        -v reportgen-output-$VOLUME_SUFFIX:/output/ \
        reportgen

docker run --rm \
        -v reportgen-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/metrics.json | python -m json.tool
        
docker volume rm reportgen-output-$VOLUME_SUFFIX
