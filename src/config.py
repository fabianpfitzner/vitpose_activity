"""
Configuration file for ViTPoseActivity.

This file centralizes all path and configuration settings, making it easy
to adapt the code for different environments (Docker, local, cluster, etc.).

Environment Variables:
    VPA_BASE_DIR: Base directory for the project (default: /workspace)
    VPA_DATA_DIR: Directory containing datasets (default: {BASE_DIR}/data)
    VPA_MODEL_DIR: Directory for saving trained models
    VPA_RESULTS_DIR: Directory for saving results
    VPA_VITPOSE_REPO: Path to easy_ViTPose repository

Example usage:
    # Docker execution (default)
    # Paths default to /workspace/...

    # Local execution
    export VPA_BASE_DIR="/home/user/vitpose_activity"
    export VPA_DATA_DIR="/home/user/datasets"
    python src/main.py
"""

import os
from pathlib import Path

# ============================================================================
# Base Directory Configuration
# ============================================================================
# Override these via environment variables for different execution environments
BASE_DIR = Path(os.getenv('VPA_BASE_DIR', '/workspace'))
DATA_DIR = Path(os.getenv('VPA_DATA_DIR', BASE_DIR / 'data'))
SRC_DIR = Path(os.getenv('VPA_SRC_DIR', BASE_DIR / 'src'))
MODEL_DIR = Path(os.getenv('VPA_MODEL_DIR', SRC_DIR / '01_train' / 'model'))
RESULTS_DIR = Path(os.getenv('VPA_RESULTS_DIR', SRC_DIR / '01_train' / 'results'))

# Path to easy_ViTPose repository
VITPOSE_REPO = Path(os.getenv('VPA_VITPOSE_REPO', BASE_DIR / 'easy_ViTPose' / 'easy_ViTPose'))

# ============================================================================
# Dataset Paths
# ============================================================================
# Training and validation data (for baseline models)
TRAIN_DATA_PATH = DATA_DIR / 'resnet_data' / 'train'
VAL_DATA_PATH = DATA_DIR / 'resnet_data' / 'val'

# Primary dataset path - adjust the dataset name as needed
DATASET_NAME = os.getenv('VPA_DATASET', 'dataset_1500')
DATASET_PATH = DATA_DIR / DATASET_NAME

# Alternative dataset paths (uncomment to use)
# DATASET_PATH = DATA_DIR / 'stanford40'

# Test dataset path
TEST_PATH = DATA_DIR / 'test_dataset'

# ============================================================================
# Model Save Paths
# ============================================================================
VITPOSE_MODEL_PATH = MODEL_DIR / 'vitposeactivity6.pt'
RESNET_MODEL_PATH = MODEL_DIR / 'resnet2.pt'
VGG_MODEL_PATH = MODEL_DIR / 'vgg2.pt'
EFFICIENTNET_MODEL_PATH = MODEL_DIR / 'efficientnet.pt'
DENSENET_MODEL_PATH = MODEL_DIR / 'densenet.pt'
SWIN_MODEL_PATH = MODEL_DIR / 'swin.pt'
MOBILENET_MODEL_PATH = MODEL_DIR / 'mobilenet.pt'

# ============================================================================
# Training Configuration
# ============================================================================
# Whether to load predefined train/val/test splits
LOAD_SPLITS = os.getenv('VPA_LOAD_SPLITS', 'false').lower() == 'true'

# Whether to save training plots and results
PLOT_RESULT = os.getenv('VPA_PLOT_RESULT', 'true').lower() == 'true'

# Model comparison flags - set to False to skip training specific models
COMPARE_RESNET = os.getenv('VPA_COMPARE_RESNET', 'true').lower() == 'true'
COMPARE_VGG = os.getenv('VPA_COMPARE_VGG', 'true').lower() == 'true'
COMPARE_DENSENET = os.getenv('VPA_COMPARE_DENSENET', 'true').lower() == 'true'
COMPARE_SWIN = os.getenv('VPA_COMPARE_SWIN', 'true').lower() == 'true'
COMPARE_MOBILENET = os.getenv('VPA_COMPARE_MOBILENET', 'true').lower() == 'true'

# ============================================================================
# Training Hyperparameters
# ============================================================================
NUM_CLASSES = int(os.getenv('VPA_NUM_CLASSES', '3'))        # Number of activity classes
NUM_EPOCHS = int(os.getenv('VPA_NUM_EPOCHS', '150'))        # Number of training epochs
LEARNING_RATE = float(os.getenv('VPA_LEARNING_RATE', '0.001'))  # Learning rate
BATCH_SIZE = int(os.getenv('VPA_BATCH_SIZE', '250'))        # Batch size

# Data split ratios (train, validation, test)
TRAIN_SPLIT = float(os.getenv('VPA_TRAIN_SPLIT', '0.70'))
VAL_SPLIT = float(os.getenv('VPA_VAL_SPLIT', '0.15'))
TEST_SPLIT = float(os.getenv('VPA_TEST_SPLIT', '0.15'))

# Validate split ratios sum to 1.0
assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"

SPLIT_RATIOS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

# ============================================================================
# ViTPose Model Configuration
# ============================================================================
# ViTPose model size: 's' (small), 'b' (base), 'l' (large), 'h' (huge)
MODEL_SIZE = os.getenv('VPA_MODEL_SIZE', 'h')

# YOLO detector size: 'n' (nano), 's' (small)
YOLO_SIZE = os.getenv('VPA_YOLO_SIZE', 's')

# Dataset format: 'coco', 'coco_25', 'wholebody', 'mpii', 'aic', 'ap10k', 'apt36k'
POSE_DATASET = os.getenv('VPA_POSE_DATASET', 'coco')

# Model repository for downloading pre-trained weights
HUGGINGFACE_REPO = os.getenv('VPA_HF_REPO', 'JunkyByte/easy_ViTPose')

# ============================================================================
# Activity Label Configuration
# ============================================================================
# Default: 3-class construction activity classification
PREDICTION_MAP = {
    'layouting': 0,
    'fixing': 1,
    'transporting': 2
}

# Alternative: 2-class binary classification
# PREDICTION_MAP = {
#     'other_activity': 0,
#     'fixing': 1
# }

# Reverse mapping for visualization
REVERSE_PREDICTION_MAP = {v: k for k, v in PREDICTION_MAP.items()}

# ============================================================================
# Data Augmentation Configuration
# ============================================================================
# Data augmentation parameters for training
AUGMENTATION_CONFIG = {
    'rotation_degrees': int(os.getenv('VPA_AUG_ROTATION', '10')),
    'translate': (
        float(os.getenv('VPA_AUG_TRANSLATE_X', '0.1')),
        float(os.getenv('VPA_AUG_TRANSLATE_Y', '0.1'))
    ),
    'scale': (
        float(os.getenv('VPA_AUG_SCALE_MIN', '0.9')),
        float(os.getenv('VPA_AUG_SCALE_MAX', '1.1'))
    )
}

# ============================================================================
# Helper Functions
# ============================================================================
def ensure_directories():
    """Create necessary directories if they don't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def print_config():
    """Print current configuration for debugging."""
    print("=" * 70)
    print("ViTPoseActivity Configuration")
    print("=" * 70)
    print(f"BASE_DIR:        {BASE_DIR}")
    print(f"DATA_DIR:        {DATA_DIR}")
    print(f"DATASET_PATH:    {DATASET_PATH}")
    print(f"MODEL_DIR:       {MODEL_DIR}")
    print(f"RESULTS_DIR:     {RESULTS_DIR}")
    print(f"VITPOSE_REPO:    {VITPOSE_REPO}")
    print("-" * 70)
    print(f"NUM_CLASSES:     {NUM_CLASSES}")
    print(f"NUM_EPOCHS:      {NUM_EPOCHS}")
    print(f"LEARNING_RATE:   {LEARNING_RATE}")
    print(f"BATCH_SIZE:      {BATCH_SIZE}")
    print(f"SPLIT_RATIOS:    {SPLIT_RATIOS}")
    print("-" * 70)
    print(f"MODEL_SIZE:      {MODEL_SIZE}")
    print(f"POSE_DATASET:    {POSE_DATASET}")
    print("-" * 70)
    print(f"COMPARE_RESNET:  {COMPARE_RESNET}")
    print(f"COMPARE_VGG:     {COMPARE_VGG}")
    print(f"COMPARE_DENSENET: {COMPARE_DENSENET}")
    print(f"COMPARE_SWIN:    {COMPARE_SWIN}")
    print(f"COMPARE_MOBILENET: {COMPARE_MOBILENET}")
    print("=" * 70)

if __name__ == '__main__':
    # Print configuration when run directly
    print_config()

    # Check if required paths exist
    print("\nPath Status:")
    print(f"DATA_DIR exists:     {DATA_DIR.exists()}")
    print(f"DATASET_PATH exists: {DATASET_PATH.exists()}")
    print(f"VITPOSE_REPO exists: {VITPOSE_REPO.exists()}")

    # Create output directories
    ensure_directories()
    print(f"\nCreated output directories:")
    print(f"  - {MODEL_DIR}")
    print(f"  - {RESULTS_DIR}")

