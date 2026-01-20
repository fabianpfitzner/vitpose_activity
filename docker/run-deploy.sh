#!/bin/bash
# Deployment script for ViTPoseActivity
#
# This script runs the inference/deployment pipeline in a Docker container with GPU support.
#
# Usage:
#   Basic: ./run-deploy.sh
#   Custom dataset: VPA_DATASET=dataset_cm10 ./run-deploy.sh
#   Custom data location: DATA_DIR=/path/to/data ./run-deploy.sh
#
# Environment Variables (optional):
#   DATA_DIR - Path to your data directory on host (default: ~/vitpose_activity_data)
#   CODE_DIR - Path to this repository (default: parent of docker/)
#   VPA_DATASET - Dataset folder name inside DATA_DIR (default: dataset_1500)

# Configure paths
DATA_DIR="${DATA_DIR:-${HOME}/vitpose_activity_data}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "=================================================="
echo "ViTPoseActivity Deployment"
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

# Run deployment in Docker container
docker run --name worker_activity --gpus all \
  -p 8889:8889 \
  $DOCKER_ENV_VARS \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CODE_DIR}:/workspace/src" \
  -t worker_activity_img \
  python3 /workspace/src/03_deploy/main.py
