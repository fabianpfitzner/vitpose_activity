# Docker Setup for ViTPoseActivity

This directory contains Docker configuration files for building and running ViTPoseActivity in a containerized environment.

## Files

- **`dockerfile`** - Docker image definition with all dependencies
- **`build_image.sh`** - Script to build the Docker image
- **`run-train.sh`** - Script to run training in Docker
- **`run-deploy.sh`** - Script to run inference/deployment in Docker

## Quick Start

### 1. Build the Docker Image

```bash
cd docker
./build_image.sh
```

This will create a Docker image named `worker_activity_img` with:
- PyTorch 2.1.0
- CUDA 11.8
- All required dependencies
- easy_ViTPose repository

### 2. Prepare Your Data

Organize your dataset as follows:

```
~/vitpose_activity_data/
├── dataset_1500/
│   ├── layouting/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── fixing/
│   │   └── ...
│   └── transporting/
│       └── ...
├── test_dataset/
│   └── ...
└── [Downloaded model weights will appear here]
```

### 3. Run Training

```bash
cd docker
./run-train.sh
```

**Note**: By default, this looks for data in `~/vitpose_activity_data/dataset_1500`. 

To use a different data location:
```bash
DATA_DIR="/path/to/your/data" ./run-train.sh
```

To use a different dataset folder:
```bash
VPA_DATASET="dataset_cm10" ./run-train.sh
```

Combine both:
```bash
DATA_DIR="/path/to/data" VPA_DATASET="dataset_cm10" ./run-train.sh
```

### 4. Run Deployment/Inference

```bash
cd docker
./run-deploy.sh
```

## Configuration

### Environment Variables

You can customize the host machine data directory location and dataset selection:

```bash
# Set custom data directory (on your host machine)
export DATA_DIR="/path/to/your/data"

# Set custom dataset folder name
export VPA_DATASET="dataset_cm10"

# Run with custom settings
./run-train.sh
```

**Important**: 
- `DATA_DIR` specifies where your data is located on the **host machine**
- `VPA_DATASET` specifies the dataset folder name inside `DATA_DIR` (default: `dataset_1500`)
- Do NOT modify paths in `src/config.py` - those are internal Docker paths (`/workspace/...`)
- The `DATA_DIR` location is mounted into the container at `/workspace/data`
- Your dataset will be accessible at `/workspace/data/${VPA_DATASET}` inside the container

### Docker Run Options

The scripts use the following Docker options:
- `--gpus all` - Use all available GPUs
- `-p 8889:8889` - Expose port 8889 (for Jupyter notebooks if needed)
- `-e VPA_DATA_DIR=/workspace/data` - Set data directory inside container (internal path)
- `-v ${DATA_DIR}:/workspace/data` - Mount host data directory to container
- `-v ${CODE_DIR}:/workspace/src` - Mount source code

### Modifying Training Parameters

To change training parameters, pass environment variables to the run scripts:

```bash
VPA_NUM_EPOCHS=200 VPA_BATCH_SIZE=128 VPA_LEARNING_RATE=0.0005 ./run-train.sh
```

Or manually run Docker with custom parameters:

```bash
docker run --name worker_activity --gpus all \
  -e VPA_DATA_DIR=/workspace/data \
  -e VPA_NUM_EPOCHS=200 \
  -e VPA_BATCH_SIZE=128 \
  -e VPA_LEARNING_RATE=0.0005 \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CODE_DIR}:/workspace/src" \
  -t worker_activity_img \
  python3 /workspace/src/main.py
```

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Docker runtime is installed:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Permission Errors

If you encounter permission errors with mounted volumes:
```bash
# Add user ID mapping
docker run --user $(id -u):$(id -g) ...
```

### Out of Memory

Reduce batch size in `src/config.py` or via environment variable:
```bash
docker run -e VPA_BATCH_SIZE=64 ...
```

## Advanced Usage

### Interactive Shell

To explore the container interactively:

```bash
docker run -it --rm --gpus all \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CODE_DIR}:/workspace/src" \
  worker_activity_img \
  /bin/bash
```

### Jupyter Notebook

To run Jupyter inside the container:

```bash
docker run --name jupyter_vitpose --gpus all \
  -p 8889:8889 \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CODE_DIR}:/workspace/src" \
  -t worker_activity_img \
  jupyter notebook --ip=0.0.0.0 --port=8889 --allow-root
```

Then access at `http://localhost:8889`

### Custom Python Script

To run a custom script:

```bash
docker run --rm --gpus all \
  -v "${DATA_DIR}:/workspace/data" \
  -v "${CODE_DIR}:/workspace/src" \
  worker_activity_img \
  python3 /workspace/src/your_script.py
```

## System Requirements

- Docker Engine 20.10+
- NVIDIA Docker Runtime
- NVIDIA GPU with CUDA support
- At least 16GB GPU memory (recommended)
- At least 50GB disk space for image and data

## See Also

- Main README: `../README.md`
- Configuration guide: `../src/config.py`
- Release checklist: `../RELEASE_CHECKLIST.md`

