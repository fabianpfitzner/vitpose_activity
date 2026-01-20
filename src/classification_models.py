from pickle import GLOBAL

"""
Baseline classification models for activity recognition comparison.

This module provides training and evaluation functions for several
pre-trained CNN and transformer models used as baselines to compare
against ViTPoseActivity:
- ResNet-152
- VGG-19
- DenseNet-169
- Swin Transformer (Small)
- MobileNetV3-Large

All models use transfer learning with ImageNet pre-trained weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True
plt.ion()

# Global variables to store training metrics
# NOTE: These are reset at the start of each model's training
val_accuracies = []      # Validation accuracies per epoch
epochs_eval = []         # Epoch numbers when validation was performed
test_accuracies = []     # Test accuracies (if applicable)

# ============================================================================
# Data Transformation Pipelines
# ============================================================================
# Standard ImageNet preprocessing with data augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Alternative: simpler transforms without normalization (commented for reference)
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ToTensor()
#     ]),
#     'val': None,
#     'test': None
# }

# Global variables for datasets and loaders (set by prepare_resnet_data)
dataset_sizes = None
dataloaders = None


def count_trainable_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_custom_datasets(images_preds_train, images_preds_val, images_preds_test, data_transforms):
    """
    Create custom PyTorch datasets from image-label mappings.

    Args:
        images_preds_train (dict): Training set image paths to labels
        images_preds_val (dict): Validation set image paths to labels
        images_preds_test (dict): Test set image paths to labels
        data_transforms (dict): Transformations for train/val/test

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = CustomDataset(images_preds_train, data_transforms['train'])
    val_dataset = CustomDataset(images_preds_val, data_transforms['val'])
    test_dataset = CustomDataset(images_preds_test, data_transforms['test'])
    return train_dataset, val_dataset, test_dataset


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images from file paths with associated labels.
    """
    def __init__(self, data, transform=None):
        """
        Args:
            data (dict): Dictionary mapping image paths to integer labels
            transform (callable, optional): Transform to apply to images
        """
        self.data = list(data.items())
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def prepare_resnet_data(images_preds_train, images_preds_val, image_preds_test, batch_size):
    """
    Prepare data loaders for baseline model training.

    Sets global dataloaders and dataset_sizes variables.

    Args:
        images_preds_train (dict): Training set image-to-label mapping
        images_preds_val (dict): Validation set image-to-label mapping
        image_preds_test (dict): Test set image-to-label mapping
        batch_size (int): Batch size for data loaders
    """
    global dataloaders,dataset_sizes
    # Create custom datasets
    train_dataset, val_dataset, test_dataset = create_custom_datasets(images_preds_train, images_preds_val,
                                                              image_preds_test, data_transforms)

    # Dataset sizes
    dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)  # Using 'val' transforms for test
    }

    # Create data loaders
    dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }



def train_model(model, criterion, optimizer, num_epochs=25):
    """
    Train a classification model with validation.

    Uses global dataloaders and dataset_sizes. Updates global val_accuracies
    and epochs_eval lists during training.

    Args:
        model (torch.nn.Module): Model to train
        criterion: Loss function
        optimizer: Optimizer for training
        num_epochs (int): Number of training epochs. Default: 25

    Returns:
        torch.nn.Module: Trained model with best validation weights loaded
    """
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, f'best_model_{model.__class__.__name__}.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            if epoch == 0:
                epoch_acc = torch.tensor(0.0, dtype=torch.double).to(device)
                val_accuracies.append(epoch_acc)
                epochs_eval.append(epoch)
                continue

            print(f'Epoch {epoch}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    # Store true and predicted labels for each phase
                    true_labels = labels.data.cpu().numpy()
                    preds_labels = preds.cpu().numpy()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if phase == 'val':
                    val_accuracies.append(epoch_acc)
                    epochs_eval.append(epoch)
                if phase == 'test':
                    test_accuracies.append(epoch_acc)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def test_model(model, print_report=True, device='cuda'):
    """
    Evaluate a trained model on the test set.

    Uses global dataloaders['test'] and dataset_sizes['test'].

    Args:
        model (torch.nn.Module): Trained model to evaluate
        print_report (bool): Whether to print classification report. Default: True
        device (str): Device to run evaluation on. Default: 'cuda'

    Returns:
        tuple: (all_labels, all_preds, accuracy)
    """
    corrects = 0
    all_labels = []
    all_preds = []

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)

            # Accumulate labels and predictions
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    # Calculate accuracy
    accuracy = corrects.double() / dataset_sizes['test']

    print(f'accuracy {accuracy * 100:.2f}%')

    if print_report:
        print(classification_report(all_labels, all_preds))

    return all_labels, all_preds, accuracy



def train_resnet(epochs, num_classes):
    global val_accuracies, epochs_eval, test_accuracies
    val_accuracies = []
    epochs_eval = []
    test_accuracies = []
    model_ft = models.resnet152(weights='IMAGENET1K_V1')  # 'IMAGENET1K_V1'
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, num_classes) #or 3

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           num_epochs=epochs)

    return model_ft, epochs_eval, val_accuracies

def train_vgg(epochs, num_classes):
    global val_accuracies, epochs_eval, test_accuracies
    model_ft = models.vgg19(weights='IMAGENET1K_V1')  # 'IMAGENET1K_V1'
    # num_ftrs = model_ft.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    # model_ft.fc = nn.Linear(num_ftrs, 3)
    # Newly created modules have require_grad=True by default
    num_features = model_ft.classifier[6].in_features
    features = list(model_ft.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)]) # or 3
    model_ft.classifier = nn.Sequential(*features)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    val_accuracies = []
    epochs_eval = []
    test_accuracies = []

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           num_epochs=epochs)

    return model_ft, epochs_eval, val_accuracies



def train_densenet(epochs, num_classes):
    global val_accuracies, epochs_eval, test_accuracies
    val_accuracies = []
    epochs_eval = []
    test_accuracies = []
    model_ft = models.densenet169(weights='IMAGENET1K_V1')  # Pretrained weights
    num_ftrs = model_ft.classifier.in_features  # Get input features of the last layer

    # Replace the last layer to match the number of classes
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)

    val_accuracies = []
    epochs_eval = []
    test_accuracies = []

    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=epochs)

    return model_ft, epochs_eval, val_accuracies


def train_swin(epochs, num_classes):
    global val_accuracies, epochs_eval, test_accuracies
    val_accuracies = []
    epochs_eval = []
    test_accuracies = []
    model_ft = models.swin_s(weights='IMAGENET1K_V1')  # Pretrained weights
    num_ftrs = model_ft.head.in_features  # Get input features of the last layer

    # Replace the last layer to match the number of classes
    model_ft.head = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)

    val_accuracies = []
    epochs_eval = []
    test_accuracies = []

    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=epochs)

    return model_ft, epochs_eval, val_accuracies


def train_mobilenet(epochs, num_classes):
    global val_accuracies, epochs_eval, test_accuracies
    val_accuracies = []
    epochs_eval = []
    test_accuracies = []
    model_ft = models.mobilenet_v3_large(weights='IMAGENET1K_V1')  # Pretrained weights
    num_ftrs = model_ft.classifier[3].in_features  # Get input features of the last layer

    # Replace the last layer to match the number of classes
    model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)

    val_accuracies = []
    epochs_eval = []
    test_accuracies = []

    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=epochs)

    return model_ft, epochs_eval, val_accuracies

