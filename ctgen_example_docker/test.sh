#!/usr/bin/env bash

# This script runs the GenerateCT container and then exports the generated
# .mha image files to a local directory named 'exported_images'
# for visualization.

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# --- 1. Build the Docker Image ---
# (This part is unchanged)
./build.sh

# --- 2. Prepare for the Run ---
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"
EXPORT_DIR="$SCRIPTPATH/exported_images"

# Create a clean directory on your local machine to receive the images
echo "Creating export directory at: $EXPORT_DIR"
mkdir -p "$EXPORT_DIR"
rm -f "$EXPORT_DIR"/*

# Create the named Docker volume where the container will write its output
docker volume create generatect-output-$VOLUME_SUFFIX

# --- 3. Run the Algorithm Container ---
# This command runs your algorithm. It saves its output to the named volume.
# (This part is unchanged)
echo "Running the main algorithm container..."
docker run --rm\
        --gpus '"device=1"'  \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="bridge" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v "$SCRIPTPATH/test/":/input/ \
        -v "generatect-output-$VOLUME_SUFFIX":/output/ \
        generatect
echo "Algorithm run complete."


# --- 4. EXPORT THE IMAGES ---
# This is the new, essential step.
# We start a small, temporary 'alpine' container.
# - Mount the Docker volume (containing our .mha files) to /data.
# - Mount our local export directory to /export_destination.
# - Run a command to copy everything from /data to /export_destination.
echo "Exporting images from volume to local directory..."
docker run --rm \
        -v "generatect-output-$VOLUME_SUFFIX":/data/ \
        -v "$EXPORT_DIR":/export_destination/ \
        alpine:latest \
        cp -a /data/. /export_destination/

echo "Successfully exported images."
ls -l "$EXPORT_DIR"


# --- 5. Clean Up ---
# (This part is unchanged)
# The script now deletes the named volume, but that's okay because we
# have already copied the files we need.
echo "Cleaning up Docker volume..."
docker volume rm "generatect-output-$VOLUME_SUFFIX"

echo "✅ Done. Your images are ready for visualization in the '$EXPORT_DIR' folder."