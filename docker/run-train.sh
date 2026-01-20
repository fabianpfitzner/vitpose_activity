#!/bin/bash
# Training script for ViTPoseActivity
#
# This script runs the training pipeline in a Docker container with GPU support.
#
# Usage:
#   Basic: ./run-train.sh
#   Custom dataset: VPA_DATASET=dataset_cm10 ./run-train.sh
#   Custom data location: DATA_DIR=/path/to/data ./run-train.sh
#   Multiple params: VPA_DATASET=dataset_cm10 VPA_NUM_EPOCHS=200 ./run-train.sh
#
# Environment Variables (optional):
#   DATA_DIR - Path to your data directory on host (default: ~/vitpose_activity_data)
#   CODE_DIR - Path to this repository (default: parent of docker/)
#   VPA_DATASET - Dataset folder name inside DATA_DIR (default: dataset_1500)
#   VPA_NUM_EPOCHS - Number of training epochs (default: 100)
#   VPA_BATCH_SIZE - Batch size (default: 128)
#   VPA_LEARNING_RATE - Learning rate (default: 0.0005)
#   VPA_LOAD_SPLITS - Load predefined splits (default: false)
#   VPA_PLOT_RESULT - Save training plots (default: true)

# Configure paths
DATA_DIR="${DATA_DIR:-${HOME}/vitpose_activity_data}"
CODE_DIR="${CODE_DIR:-$(pwd)/src}"

echo "=================================================="
echo "ViTPoseActivity Training"
echo "=================================================="
echo "Data directory: ${DATA_DIR}"
echo "Code directory: ${CODE_DIR}"
echo "Dataset name: ${VPA_DATASET:-dataset_1500 (default)}"
echo "=================================================="

# Remove existing container if it exists
docker rm worker_activity 2>/dev/null

# Build docker run command with environment variables
DOCKER_ENV_VARS="-e VPA_DATA_DIR=/workspace/data -e VPA_BASE_DIR=/workspace"

# Pass through configuration environment variables if set
[ -n "$VPA_DATASET" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_DATASET=$VPA_DATASET"
[ -n "$VPA_NUM_EPOCHS" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_NUM_EPOCHS=$VPA_NUM_EPOCHS"
[ -n "$VPA_BATCH_SIZE" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_BATCH_SIZE=$VPA_BATCH_SIZE"
[ -n "$VPA_LEARNING_RATE" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_LEARNING_RATE=$VPA_LEARNING_RATE"
[ -n "$VPA_LOAD_SPLITS" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_LOAD_SPLITS=$VPA_LOAD_SPLITS"
[ -n "$VPA_PLOT_RESULT" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_PLOT_RESULT=$VPA_PLOT_RESULT"
[ -n "$VPA_COMPARE_RESNET" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_COMPARE_RESNET=$VPA_COMPARE_RESNET"
[ -n "$VPA_COMPARE_VGG" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_COMPARE_VGG=$VPA_COMPARE_VGG"
[ -n "$VPA_COMPARE_DENSENET" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_COMPARE_DENSENET=$VPA_COMPARE_DENSENET"
[ -n "$VPA_COMPARE_SWIN" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_COMPARE_SWIN=$VPA_COMPARE_SWIN"
[ -n "$VPA_COMPARE_MOBILENET" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VPA_COMPARE_MOBILENET=$VPA_COMPARE_MOBILENET"

# Run training in Docker container
docker run --name worker_activity --gpus all \
  -p 8889:8889 \
  $DOCKER_ENV_VARS \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CODE_DIR}:/workspace/src" \
  -t worker_activity_img \
  python3 /workspace/src/main.py
