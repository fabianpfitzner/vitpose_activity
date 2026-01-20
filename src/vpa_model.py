"""
ViTPoseActivity model training and data loading utilities.

This module provides the main ViTPoseActivity class for training and evaluation,
along with data loading, preprocessing, and utility functions for activity
recognition from construction site images.

Configuration is managed via config.py, which supports environment variables
for easy customization across different execution environments.
"""

import os
import sys

# ============================================================================
# Import Configuration and Setup Path
# ============================================================================
# IMPORTANT: Must import config and add VITPOSE_REPO to sys.path BEFORE
# importing any easy_ViTPose modules (configs.*, vit_utils.*)
from config import (
    VITPOSE_REPO,
    DATA_DIR,
    PREDICTION_MAP,
    REVERSE_PREDICTION_MAP,
    MODEL_SIZE,
    YOLO_SIZE,
    POSE_DATASET,
    HUGGINGFACE_REPO
)

# Add ViTPose repository to path (required for Docker environment)
sys.path.append(str(VITPOSE_REPO))

# ============================================================================
# Standard Library and Third-Party Imports
# ============================================================================
import pandas as pd
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
from tempfile import TemporaryDirectory
from torchvision import datasets, models, transforms

# ============================================================================
# easy_ViTPose Imports (require sys.path.append above)
# ============================================================================
from configs.ViTPose_common import data_cfg
from vit_utils.inference import draw_bboxes, pad_image
from vit_utils.util import dyn_model_import

# ============================================================================
# Local Project Imports
# ============================================================================
from vpa_model_architecture import ViTPoseClassifier
from helpers import load_np_image

# ============================================================================
# COCO Skeleton Definition (for reference)
# ============================================================================
# COCO format defines 17 keypoints with the following structure:
# Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
# Skeleton: 19 connections between keypoints defining the body structure
#
# Full COCO skeleton structure:
# {
#     "keypoints": {
#         0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
#         5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
#         9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
#         13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
#     },
#     "skeleton": [
#         [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
#         [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1],
#         [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
#     ]
# }

# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Body part groupings for analysis (indices into COCO keypoints)
upper_part = [0, 1, 2, 3, 4, 5, 6]      # Head and shoulders
middle_part = [7, 8, 9, 10, 11, 12]      # Arms and hips
lower_part = [13, 14, 15, 16]            # Legs

# ============================================================================
# Model Configuration
# ============================================================================
# Model file extensions
ext = '.pth'         # Model file extension
ext_yolo = '.pt'     # YOLO file extension

MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = HUGGINGFACE_REPO
FILENAME = os.path.join(MODEL_TYPE, f'{POSE_DATASET}/vitpose-' + MODEL_SIZE + f'-{POSE_DATASET}') + ext
FILENAME_YOLO = 'yolov8/yolov8' + YOLO_SIZE + ext_yolo

model_path = None
yolo_path = None

# ============================================================================
# Activity Label Mappings
# ============================================================================
# Activity label mappings are now imported from config.py
# Use PREDICTION_MAP and REVERSE_PREDICTION_MAP
prediction_map = PREDICTION_MAP
reverse_prediction_map = REVERSE_PREDICTION_MAP

# ============================================================================
# Model Path Discovery
# ============================================================================
# Search for pre-trained model files in the data directory
for root, dirs, files in os.walk(str(DATA_DIR)):
    for file in files:
        if file.endswith('.pth') or file.endswith('.pt'):
            if 'vitpose' in file:
                model_path = os.path.join(root, file)
            else:
                yolo_path = os.path.join(root, file)

# Download models from HuggingFace if not found locally
if model_path is None:
    print(f'Downloading model {REPO_ID}/{FILENAME}')
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(DATA_DIR),
                                 local_dir_use_symlinks=False)

if yolo_path is None:
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO, local_dir=str(DATA_DIR),
                                local_dir_use_symlinks=False)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_data(data_path, data_transforms=None, predefined_split=None, split=(0.7, 0.15, 0.15), batch_size=1):
    """
    Load and prepare training, validation, and test datasets.

    Args:
        data_path (str): Path to the dataset directory
        data_transforms (dict, optional): Transformations for train/val/test sets
        predefined_split (tuple, optional): Predefined data splits to use
        split (tuple): Train/val/test split ratios. Default: (0.7, 0.15, 0.15)
        batch_size (int): Batch size for data loaders. Default: 1

    Returns:
        tuple: (dataloaders, dataset_sizes, images_preds_train, images_preds_val, images_preds_test)
    """
    # Get data
    images_preds_train, images_preds_val, images_preds_test = get_images_prediction(data_path, split=split)
    if predefined_split is not None:
        images_preds_train, images_preds_val, images_preds_test = predefined_split

    training_data = ConstructionPoseDataset(images_preds_train,
                                            None if data_transforms is None else data_transforms['train'])
    validation_data = ConstructionPoseDataset(images_preds_val,
                                              None if data_transforms is None else data_transforms['val'])
    test_data = ConstructionPoseDataset(images_preds_test, None if data_transforms is None else data_transforms['test'])

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    dataset_sizes = {x: len({'train': training_data, 'val': validation_data, 'test': test_data}[x]) for x in
                    ['train', 'val', 'test']}

    return {'train': train_dataloader, 'val': val_dataloader,
            'test': test_dataloader}, dataset_sizes, images_preds_train, images_preds_val, images_preds_test


def save_splits(images_preds_train, images_preds_val, path='temp_split_train.csv'):
    """
    Save dataset splits to CSV files for reproducibility.

    Args:
        images_preds_train (dict): Training set image-to-label mapping
        images_preds_val (dict): Validation set image-to-label mapping
        path (str): Path to save training split CSV. Default: 'temp_split_train.csv'
    """
    for x in [images_preds_train, images_preds_val]:
        df = pd.DataFrame(x.items(), columns=['name', 'prediction'])
        df.to_csv(path)
        path = path.replace('train', 'val')


def load_splits(path='temp_split_train.csv'):
    """
    Load previously saved dataset splits from CSV files.

    Args:
        path (str): Path to training split CSV. Default: 'temp_split_train.csv'

    Returns:
        list: List of two dicts (train and val) with image-to-label mappings
    """
    splits = []
    for _ in range(2):
        df = pd.read_csv(path, names=['name', 'prediction'], dtype={'name': 'string', 'prediction': 'int'}, skiprows=1)
        splits.append(dict(zip(df['name'], df['prediction'])))
        path = path.replace('train', 'val')

    return splits


def load_test_data(data_path):
    """
    Load test dataset without splitting.

    Args:
        data_path (str): Path to test dataset directory

    Returns:
        tuple: (dataset, image_predictions)
    """
    images_preds, _ = get_images_prediction(data_path, 1.0)
    return generate_input_output_tensor(images_preds), images_preds


# ============================================================================
# Dataset Classes
# ============================================================================

class ConstructionDataset(Dataset):
    """
    Dataset class for pre-processed construction activity data.

    Stores tensors with associated image configuration metadata.
    """
    def __init__(self, X, y, img_conf):
        self.X = X
        self.y = y
        self.org_h = img_conf['org_h']
        self.org_w = img_conf['org_w']
        self.pad = img_conf['pad']
        self.x_y = img_conf['x_y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], {
            'org_h': self.org_h[idx],
            'org_w': self.org_w[idx],
            'pad': self.pad[idx],
            'x_y': self.x_y[idx]
        }


class ConstructionPoseDataset(Dataset):
    """
    Dataset class for construction activity images with on-the-fly preprocessing.

    Loads images from disk and applies transformations during training.
    """
    def __init__(self, data, transform=None):
        """
        Args:
            data (dict): Dictionary mapping image paths to labels
            transform (callable, optional): Transform to apply to images
        """
        self.data = list(data.items())
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        # Load image

        image = Image.open(image_path)
        if self.transform is not None:
            # Convert image to numpy array
            img = image.convert('RGB')

            # Apply transform to numpy array
            tensor_transformed = self.transform(img)

            # Convert transformed numpy array back to PIL image
            image = transforms.ToPILImage()(tensor_transformed)

        img = np.array(image, dtype=np.uint8)

        # Apply transform
        img_input, org_h, org_w, pad, x_y = pre_img(img)

        return img_input, org_h, org_w, pad, x_y, label


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.best_score = -val_loss


def create_confusion_mtx(y_gt, y_predicted, save_path=None):
    y_gt_names = [reverse_prediction_map[x] for x in y_gt]
    y_predicted_names = [reverse_prediction_map[x] for x in y_predicted]
    cf_mtx = confusion_matrix(y_gt_names, y_predicted_names, labels=list(prediction_map.keys()))

    plt.figure(figsize=(7, 5))
    sn.heatmap(cf_mtx, cmap="crest", fmt="d", xticklabels=list(prediction_map.keys()),
               yticklabels=list(prediction_map.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    print('Confusion Mtx')
    # Customizing annotations for each cell
    for i in range(len(cf_mtx)):
        for j in range(len(cf_mtx[i])):
            plt.text(j + 0.5, i + 0.5, cf_mtx[i][j], ha='center', va='center', color='black', fontsize=14)

    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)


def generate_input_output_tensor(image_predictions, device='cuda'):
    img_inputs = []
    img_conf = {}
    img_conf['org_h'] = []
    img_conf['org_w'] = []
    img_conf['pad'] = []
    img_conf['x_y'] = []

    imgs, predictions = shuffle_items(image_predictions)
    for image_path in imgs:
        img = load_np_image(full_img_path=image_path)
        img_input, org_h, org_w, pad, x_y = pre_img(img)
        img_inputs.append(img_input)
        img_conf['org_h'].append(org_h)
        img_conf['org_w'].append(org_w)
        img_conf['pad'].append(pad)
        img_conf['x_y'].append(x_y)

    for k, v in img_conf.items():
        img_conf[k] = torch.from_numpy(np.array(v)).to(torch.device(device))

    # img_conf = TensorDict(img_conf, batch_size=[len(list(img_conf['org_h']))])
    X_tensor = torch.from_numpy(np.array(img_inputs)).to(torch.device(device))
    y_tensor = torch.tensor(predictions, dtype=torch.long).to(torch.device(device))
    return ConstructionDataset(X_tensor, y_tensor, img_conf)


def shuffle_items(image_predictions):
    imgs = list(image_predictions.keys())
    random.shuffle(imgs)
    predictions = []
    for img in imgs:
        predictions.append(image_predictions[img])

    return imgs, predictions


def get_images_from_dir(img_dir):
    """
    Get list of valid image files from a directory, excluding hidden files.

    Args:
        img_dir: Directory path containing images

    Returns:
        List of full paths to image files
    """
    imgs = []
    for img in os.listdir(img_dir):
        # Skip hidden files (starting with ".") and non-image files
        if img.startswith('.') or not img.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        imgs.append(os.path.join(img_dir, img))

    return imgs


def get_images_prediction(img_dir, split=(0.7, 0.15, 0.15)):
    """
    Splits images into training, validation, and testing datasets.

    Args:
        img_dir (str): Directory containing subdirectories of images.
        split (list): A list containing three percentages (e.g., [0.7, 0.15, 0.15])
                      for training, validation, and testing splits.

    Returns:
        tuple: Three dictionaries containing image-to-label mappings for training,
               validation, and testing datasets.
    """
    if sum(split) != 1.0:
        raise ValueError("Split percentages must sum to 1.0")

    image_predictions_train = {}
    image_predictions_val = {}
    image_predictions_test = {}

    # Iterate through subdirectories
    for sub_dir in os.listdir(img_dir):
        # Skip hidden files/directories (starting with ".")
        if sub_dir.startswith('.'):
            continue

        full_path = os.path.join(img_dir, sub_dir)
        if not os.path.isdir(full_path):
            continue

        # Get images from the directory
        imgs = get_images_from_dir(full_path)  # Assuming this function exists
        random.shuffle(imgs)  # Shuffle the images randomly

        prediction = os.path.basename(sub_dir)  # Assuming subdir name is the label
        if prediction not in prediction_map:
            raise KeyError(f"Prediction '{prediction}' not found in prediction_map")

        train_thres = int(split[0] * len(imgs))
        val_thres = train_thres + int(split[1] * len(imgs))

        for idx, img in enumerate(imgs):
            if idx < train_thres:
                image_predictions_train[img] = prediction_map[prediction]
            elif idx < val_thres:
                image_predictions_val[img] = prediction_map[prediction]
            else:
                image_predictions_test[img] = prediction_map[prediction]

    return image_predictions_train, image_predictions_val, image_predictions_test


def pre_img(img):
    img_conf = {}
    # Define the entire image as the outer frame
    x, y, w, h = 0, 0, img.shape[1], img.shape[0]

    # Calculate the center coordinates
    x_center = x + w / 2
    y_center = y + h / 2

    # Crop image and pad to 3/4 aspect ratio
    img = img[y:y + h, x:x + w]
    img, (left_pad, top_pad) = pad_image(img, 3 / 4)

    org_h, org_w = img.shape[:2]
    img_input = cv2.resize(img, data_cfg['image_size'], interpolation=cv2.INTER_LINEAR) / 255
    img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    # Convert lists to NumPy arrays before performing the subtraction
    x_y = np.array([x, y][::-1])
    pad = np.array([top_pad, left_pad])

    return img_input[0], org_h, org_w, pad, x_y


class ViTPoseActivity:
    train_losses = []
    val_accuracies = []
    test_accuracies = []
    epochs_eval = []
    val_losses = []
    epochs = 0

    def __init__(self, backbone_path=model_path, num_classes=3):
        model_cfg = dyn_model_import(POSE_DATASET, MODEL_SIZE)
        self.model = ViTPoseClassifier(cfg=model_cfg, num_classes=num_classes, pre_trained_backbone_path=backbone_path)

    def save_state(self, path):
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_losses[-1],
        }, path)

    def train(self, dataloaders, dataset_sizes, save_path=None, eval_epoch_step=50, epochs=300,
              criterion=nn.CrossEntropyLoss(),
              learning_rate=0.001,
              weight_decay=1e-4,
              device='cuda', early_stopping_patience=10, early_stop=False, shuffle=False):
        self.model.to(torch.device(device))
        since = time.time()
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        # reset
        self.train_losses = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.epochs_eval = []
        self.val_losses = []
        self.epochs = epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(epochs):
                if epoch == 0:
                    epoch_acc = torch.tensor(0.0, dtype=torch.double).to(torch.device(device))
                    self.val_accuracies.append(epoch_acc)
                    self.epochs_eval.append(epoch)
                    continue

                print(f'Epoch {epoch}/{epochs}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for X, org_h, org_w, pad, x_y, label in dataloaders[phase]:
                        # zero the parameter gradients
                        self.optimizer.zero_grad()
                        X = X.to(torch.device(device))
                        org_h = org_h.to(torch.device(device))
                        org_w = org_w.to(torch.device(device))
                        pad = pad.to(torch.device(device))
                        x_y = x_y.to(torch.device(device))
                        label = label.to(torch.device(device))

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            outputs = self.model.forward(X, {'org_h': org_h, 'org_w': org_w, 'pad': pad, 'x_y': x_y})
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, label)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * X.size(0)
                        running_corrects += torch.sum(preds == label.data)

                    #if phase == 'train':
                    #scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    if phase == 'val':
                        self.val_accuracies.append(epoch_acc)
                        self.epochs_eval.append(epoch)
                        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    if phase == 'test':
                        self.test_accuracies.append(epoch_acc)

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val acc: {best_acc:4f}')

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path))

    def test(self, test_loader, dataset_size_testing, print_report=True, device='cuda'):
        self.model.eval()
        corrects = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for X, org_h, org_w, pad, x_y, label in test_loader:
                X = X.to(torch.device(device))
                org_h = org_h.to(torch.device(device))
                org_w = org_w.to(torch.device(device))
                pad = pad.to(torch.device(device))
                x_y = x_y.to(torch.device(device))
                label = label.to(torch.device(device))

                outputs = self.model.forward(X, {'org_h': org_h, 'org_w': org_w, 'pad': pad, 'x_y': x_y})
                _, preds = torch.max(outputs, 1)

                corrects += torch.sum(preds == label.data)

                # Accumulate labels and predictions
                all_labels.extend(label.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())

        accuracy = corrects.double() / dataset_size_testing

        print(f'Test accuracy {accuracy * 100:.2f}%')

        if print_report:
            print(classification_report(all_labels, all_preds))

        return all_labels, all_preds, accuracy

    def plot_train_loss(self, save_path=None):
        plt.plot(range(self.epochs), self.train_losses)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        if save_path is not None:
            plt.savefig(save_path, format='png', dpi=300)

    def plot_val_loss(self, save_path=None):
        plt.plot(self.epochs_eval, self.val_losses)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        if save_path is not None:
            plt.savefig(save_path, format='png', dpi=300)

    def plot_val_accuracy(self, save_path=None):
        plt.plot(self.epochs_eval, self.val_accuracies, color="orange")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        if save_path is not None:
            plt.savefig(save_path, format='png', dpi=300)

    def plot_test_accuracy(self, save_path=None):
        plt.plot(self.epochs_eval, self.test_accuracies, color="orange")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        if save_path is not None:
            plt.savefig(save_path, format='png', dpi=300)

    def count_trainable_parameters(self):
        """
        Counts the total number of trainable parameters in a model,
        considering only layers with requires_grad=True.

        Parameters:
        - model (torch.nn.Module): The PyTorch model.

        Returns:
        - int: Total number of trainable parameters.
        """
        # Specifically count parameters from the defined linear layers
        model = self.model
        linear_layers = [model.fc1, model.fc2]
        return sum(p.numel() for layer in linear_layers for p in layer.parameters() if p.requires_grad)
