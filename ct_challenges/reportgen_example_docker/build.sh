#!/usr/bin/env bash
# Build the BTB3D reportgen example image.
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
docker build -t reportgen-btb3d "$SCRIPTPATH"
