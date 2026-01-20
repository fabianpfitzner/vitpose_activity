"""
Helper utilities for data processing, visualization, and evaluation.

This module provides utility functions for creating confusion matrices,
plotting accuracy curves, loading images, and saving training results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from PIL import Image
import pandas as pd

# Activity label mappings for classification tasks
# Maps activity names to integer class labels
prediction_map = {'layouting': 0, 'fixing': 1, 'transporting': 2}
reverse_prediction_map = {v: k for k, v in prediction_map.items()}


def create_confusion_mtx(y_gt, y_predicted, save_path=None):
    """
    Create and visualize a normalized confusion matrix.

    Args:
        y_gt (list): Ground truth labels as integer class IDs
        y_predicted (list): Predicted labels as integer class IDs
        save_path (str, optional): Path to save the confusion matrix plot. Defaults to None.

    Returns:
        numpy.ndarray: Confusion matrix with raw counts
    """
    y_gt_names = [reverse_prediction_map[x] for x in y_gt]
    y_predicted_names = [reverse_prediction_map[x] for x in y_predicted]
    cf_mtx = confusion_matrix(y_gt_names, y_predicted_names, labels=list(prediction_map.keys()))

    # Normalize the confusion matrix
    row_sums = cf_mtx.sum(axis=1, keepdims=True)
    normalized_cf_mtx = cf_mtx / row_sums

    plt.figure(figsize=(7, 5))
    ax = sn.heatmap(normalized_cf_mtx, cmap="crest", annot=False,
                    xticklabels=list(prediction_map.keys()), yticklabels=list(prediction_map.keys()))

    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    # plt.title('Normalized Confusion Matrix')

    # Customizing annotations for each cell
    for i in range(len(cf_mtx)):
        for j in range(len(cf_mtx[i])):
            plt.text(j + 0.5, i + 0.5, f"{normalized_cf_mtx[i][j]:.2f}",
                     ha='center', va='center', color='black', fontsize=15)

    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)

    return cf_mtx


def plot_accuracy(epochs_eval, accuracies, label, color=None, save_path=None):
    """
    Plot accuracy curve over training epochs.

    Args:
        epochs_eval (list): List of epoch numbers
        accuracies (list): Corresponding accuracy values for each epoch
        label (str): Label for the plot legend
        color (str, optional): Color for the plot line. Defaults to None.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.plot(epochs_eval, accuracies, label=label, color=color)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="lower right")
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)


def convert_tensors_to_values(tensor_list):
    """
    Convert a list of PyTorch tensors to a NumPy array of float values.

    Args:
        tensor_list (list): List of PyTorch tensors

    Returns:
        numpy.ndarray: Array of converted float values
    """
    new_list = []
    for tensor in tensor_list:
        new_list.append(float(tensor.detach().cpu().numpy()))

    return np.array(new_list)


def is_jpg_image(img_path):
    """
    Check if a file path points to a JPG image.

    Args:
        img_path (str): Path to the image file

    Returns:
        bool: True if the file extension is 'jpg', False otherwise
    """
    split_path = img_path.split('.')
    return len(split_path) > 1 and split_path[1] == 'jpg'


def load_np_image(full_img_path):
    """
    Load an image as a NumPy array.

    Args:
        full_img_path (str): Full path to the image file

    Returns:
        numpy.ndarray: Image data as uint8 array
    """
    return np.array(Image.open(full_img_path), dtype=np.uint8)


def get_avg_pts(keypoints, part_indices):
    """
    Calculate the average position of multiple keypoints.

    Args:
        keypoints (numpy.ndarray): Array of keypoint coordinates
        part_indices (list): Indices of keypoints to average

    Returns:
        numpy.ndarray: Average position of the specified keypoints
    """
    pts = []
    for i in part_indices:
        pts.append(keypoints[i])

    return np.average(np.array(pts), axis=0)


def get_pts(keypoints, parts):
    """
    Extract specific keypoints from a keypoint array.

    Args:
        keypoints (numpy.ndarray): Array of all keypoints
        parts (list): Indices of keypoints to extract

    Returns:
        list: Selected keypoints
    """
    pts = []
    for i in parts:
        pts.append(keypoints[i])

    return pts


def save_training_results(epochs, val_accuracies_vitpose, val_accuracies_res, val_accuracies_vgg, save_path):
    """
    Save the training results as a CSV file.

    Parameters:
    - epochs (list): A list of epoch numbers.
    - val_accuracies_vitpose (list): Validation accuracies for ViTPoseActivity.
    - val_accuracies_res (list): Validation accuracies for ResNet-152.
    - val_accuracies_vgg (list): Validation accuracies for VGG-19.
    - save_path (str): Path to save the CSV file.
    """
    # Create a dictionary with the training results
    data = {
        "Epoch": epochs,
        "ViTPoseActivity_Validation_Accuracy": val_accuracies_vitpose,
        "ResNet_Validation_Accuracy": val_accuracies_res,
        "VGG_Validation_Accuracy": val_accuracies_vgg
    }

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame as a CSV file
    df.to_csv(save_path, index=False)
    print(f"Training results saved to {save_path}")


def save_all_training_results(
    epochs,
    val_accuracies_vitpose,
    val_accuracies_res,
    val_accuracies_vgg,
    val_accuracies_densenet,
    val_accuracies_swin,
    val_accuracies_mobilenet,
    save_path
):
    """
    Save the training results as a CSV file.

    Parameters:
    - epochs (list): A list of epoch numbers.
    - val_accuracies_vitpose (list): Validation accuracies for ViTPoseActivity.
    - val_accuracies_res (list): Validation accuracies for ResNet-152.
    - val_accuracies_vgg (list): Validation accuracies for VGG-19.
    - val_accuracies_efficientnet (list): Validation accuracies for EfficientNet-B4.
    - val_accuracies_densenet (list): Validation accuracies for DenseNet-201.
    - val_accuracies_swin (list): Validation accuracies for Swin-B.
    - val_accuracies_mobilenet (list): Validation accuracies for MobileNetV3-Large.
    - save_path (str): Path to save the CSV file.
    """
    # Create a dictionary with the training results
    data = {
        "Epoch": epochs,
        "ViTPoseActivity_Validation_Accuracy": val_accuracies_vitpose,
        "ResNet_Validation_Accuracy": val_accuracies_res,
        "VGG_Validation_Accuracy": val_accuracies_vgg,
        "DenseNet_Validation_Accuracy": val_accuracies_densenet,
        "Swin_Validation_Accuracy": val_accuracies_swin,
        "MobileNet_Validation_Accuracy": val_accuracies_mobilenet
    }

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame as a CSV file
    df.to_csv(save_path, index=False)
    print(f"Training results saved to {save_path}")