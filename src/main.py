"""
Main training script for ViTPoseActivity and comparison models.

This script trains the ViTPoseActivity model and compares it against
several baseline models (ResNet, VGG, DenseNet, Swin, MobileNet).
It supports data loading, model training, evaluation, and result visualization.

Configuration is managed via config.py, which supports environment variables
for easy customization across different execution environments.
"""

import os
import sys


import torch
from vpa_model import ViTPoseActivity, load_data, load_splits
from torchvision import datasets, models, transforms
from classification_models import prepare_resnet_data, train_resnet, train_vgg, test_model, train_swin, train_mobilenet, \
    train_densenet, count_trainable_parameters
from helpers import plot_accuracy, convert_tensors_to_values, save_all_training_results

# ============================================================================
# Import Configuration
# ============================================================================
# All paths and settings are now managed in config.py
# Override via environment variables (e.g., export VPA_DATA_DIR=/my/data)
from config import (
    DATASET_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_PATH,
    VITPOSE_MODEL_PATH,
    RESNET_MODEL_PATH,
    VGG_MODEL_PATH,
    EFFICIENTNET_MODEL_PATH,
    DENSENET_MODEL_PATH,
    SWIN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    RESULTS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    SPLIT_RATIOS,
    LOAD_SPLITS,
    PLOT_RESULT,
    COMPARE_RESNET,
    COMPARE_VGG,
    COMPARE_DENSENET,
    COMPARE_SWIN,
    COMPARE_MOBILENET,
    ensure_directories,
    print_config
)

# Create necessary output directories
ensure_directories()

# Convert Path objects to strings for compatibility
data_path = str(DATASET_PATH)
train_data_path = str(TRAIN_DATA_PATH)
val_data_path = str(VAL_DATA_PATH)
test_path = str(TEST_PATH)
vitpose_model_path = str(VITPOSE_MODEL_PATH)
resnet_model_path = str(RESNET_MODEL_PATH)
vgg_model_path = str(VGG_MODEL_PATH)
efficientnet_model_path = str(EFFICIENTNET_MODEL_PATH)
densenet_model_path = str(DENSENET_MODEL_PATH)
swin_model_path = str(SWIN_MODEL_PATH)
mobilenet_model_path = str(MOBILENET_MODEL_PATH)
results_path = str(RESULTS_DIR)



# ============================================================================
# Data Augmentation Configuration
# ============================================================================
# Data augmentation strategy tailored for pose-based activity recognition:
# • Noise Injection: Add slight noise to simulate imperfections in pose
#   detection while ensuring keypoints remain realistic
# • Affine Transformations: Apply mild scaling, rotation, or translation to
#   simulate natural variations in body pose
# • Color Jittering: Avoided since pose estimation ignores color information

# Transforms for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToTensor()
    ])
}



if __name__ == "__main__":
    # Print configuration for debugging
    print_config()

    # Load predefined data splits if available, otherwise split data randomly
    splits = load_splits() if LOAD_SPLITS else None

    print('\nLoading data...')
    # Load and prepare datasets with configured split ratios
    dataloaders, dataset_sizes, images_preds_train, images_preds_val, images_preds_test = load_data(
        data_path,
        data_transforms,
        predefined_split=splits,
        split=SPLIT_RATIOS,
        batch_size=BATCH_SIZE
    )

    # ========================================================================
    # Train ViTPoseActivity Model
    # ========================================================================
    print('******VITPOSEACTIVITY******')
    vitpose = ViTPoseActivity(num_classes=NUM_CLASSES)
    print('Started training...')
    vitpose.train(dataloaders, dataset_sizes, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # Evaluate on test set
    print('Testing model...')
    result = vitpose.test(dataloaders['test'], dataset_sizes['test'])

    # Report model size
    print(f'Trainable parameters: {vitpose.count_trainable_parameters()}')
    print(f'Total parameters: {count_trainable_parameters(vitpose.model)}')

    # Save trained model
    torch.save(vitpose, vitpose_model_path)

    # ========================================================================
    # Train and Evaluate Baseline Models
    # ========================================================================
    # Initialize variables to avoid undefined warnings
    val_accuracies_res = None
    epochs_eval_res = None
    val_accuracies_vgg = None
    epochs_eval_vgg = None
    val_accuracies_densenet = None
    epochs_eval_densenet = None
    val_accuracies_swin = None
    epochs_eval_swin = None
    val_accuracies_mobilenet = None
    epochs_eval_mobilenet = None

    if COMPARE_RESNET:
        print('******RESNET******')
        prepare_resnet_data(images_preds_train, images_preds_val, images_preds_test, BATCH_SIZE)
        res_net_model, epochs_eval_res, val_accuracies_res = train_resnet(NUM_EPOCHS, NUM_CLASSES)
        torch.save(res_net_model, resnet_model_path)
        test_model(res_net_model)
        print(f'Trainable parameters: {count_trainable_parameters(res_net_model)}')

    if COMPARE_VGG:
        print('******VGG******')
        prepare_resnet_data(images_preds_train, images_preds_val, images_preds_test, BATCH_SIZE)
        vgg_model, epochs_eval_vgg, val_accuracies_vgg = train_vgg(NUM_EPOCHS, NUM_CLASSES)
        torch.save(vgg_model, vgg_model_path)
        test_model(vgg_model)
        print(f'Trainable parameters: {count_trainable_parameters(vgg_model)}')

    if COMPARE_DENSENET:
        print('******DENSENET******')
        prepare_resnet_data(images_preds_train, images_preds_val, images_preds_test, BATCH_SIZE)
        densenet_model, epochs_eval_densenet, val_accuracies_densenet = train_densenet(NUM_EPOCHS, NUM_CLASSES)
        torch.save(densenet_model, densenet_model_path)
        test_model(densenet_model)
        print(f'Trainable parameters: {count_trainable_parameters(densenet_model)}')

    if COMPARE_SWIN:
        print('******SWIN******')
        prepare_resnet_data(images_preds_train, images_preds_val, images_preds_test, BATCH_SIZE)
        swin_model, epochs_eval_swin, val_accuracies_swin = train_swin(NUM_EPOCHS, NUM_CLASSES)
        torch.save(swin_model, swin_model_path)
        test_model(swin_model)
        print(f'Trainable parameters: {count_trainable_parameters(swin_model)}')

    if COMPARE_MOBILENET:
        print('******MOBILENET******')
        prepare_resnet_data(images_preds_train, images_preds_val, images_preds_test, BATCH_SIZE)
        mobilenet_model, epochs_eval_mobilenet, val_accuracies_mobilenet = train_mobilenet(NUM_EPOCHS, NUM_CLASSES)
        torch.save(mobilenet_model, mobilenet_model_path)
        test_model(mobilenet_model)
        print(f'Trainable parameters: {count_trainable_parameters(mobilenet_model)}')

    if PLOT_RESULT:
        # Save all training results to CSV
        save_all_training_results(
            epochs=vitpose.epochs_eval,
            val_accuracies_vitpose=convert_tensors_to_values(vitpose.val_accuracies),
            val_accuracies_res=convert_tensors_to_values(val_accuracies_res) if val_accuracies_res is not None else [],
            val_accuracies_vgg=convert_tensors_to_values(val_accuracies_vgg) if val_accuracies_vgg is not None else [],
            val_accuracies_densenet=convert_tensors_to_values(val_accuracies_densenet) if val_accuracies_densenet is not None else [],
            val_accuracies_swin=convert_tensors_to_values(val_accuracies_swin) if val_accuracies_swin is not None else [],
            val_accuracies_mobilenet=convert_tensors_to_values(val_accuracies_mobilenet) if val_accuracies_mobilenet is not None else [],
            save_path=results_path + '/training_validation_all_results.csv'
        )

        # Plot ViTPoseActivity results
        plot_accuracy(vitpose.epochs_eval, convert_tensors_to_values(vitpose.val_accuracies),
                      label="ViTPoseActivity", save_path=results_path + '/vitposeactivity.png')

        # Plot ResNet results
        if COMPARE_RESNET and val_accuracies_res is not None:
            plot_accuracy(epochs_eval_res, convert_tensors_to_values(val_accuracies_res), color='green',
                          label="ResNet-152", save_path=results_path + '/resnet152-pretrain.png')

        # Plot VGG results
        if COMPARE_VGG and val_accuracies_vgg is not None:
            plot_accuracy(epochs_eval_vgg, convert_tensors_to_values(val_accuracies_vgg), color='orange',
                          label="VGG-19", save_path=results_path + '/vgg19-pretrain.png')

        # Plot DenseNet results
        if COMPARE_DENSENET and val_accuracies_densenet is not None:
            plot_accuracy(
                epochs_eval_densenet,
                convert_tensors_to_values(val_accuracies_densenet),
                color='purple',
                label="DenseNet-169",
                save_path=results_path + '/densenet169-pretrain.png'
            )

        # Plot Swin results
        if COMPARE_SWIN and val_accuracies_swin is not None:
            plot_accuracy(
                epochs_eval_swin,
                convert_tensors_to_values(val_accuracies_swin),
                color='red',
                label="Swin-S",
                save_path=results_path + '/swins-pretrain.png'
            )

        # Plot MobileNet results
        if COMPARE_MOBILENET and val_accuracies_mobilenet is not None:
            plot_accuracy(
                epochs_eval_mobilenet,
                convert_tensors_to_values(val_accuracies_mobilenet),
                color='cyan',
                label="MobileNetV3-Large",
                save_path=results_path + '/mobilenetv3-large-pretrain.png'
            )
