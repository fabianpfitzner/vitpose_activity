"""
ViTPose-based Activity Classification Model Architecture.

This module defines the ViTPoseClassifier model that combines ViTPose backbone
for pose estimation with custom classification layers for activity recognition.
The model extracts pose keypoints and skeletal features, then classifies activities
based on body posture and joint relationships.
"""

import torch.nn as nn
import torch.nn.functional as F
import math

from vit_models.backbone.vit import ViT
from vit_models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from vit_utils.top_down_eval import keypoints_from_heatmaps

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler

# ImageNet normalization constants used for ViTPose preprocessing
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Mapping from detection categories to YOLO class IDs
# Used for object detection and filtering
DETC_TO_YOLO_YOLOC = {
    'human': [0],
    'cat': [15],
    'dog': [16],
    'horse': [17],
    'sheep': [18],
    'cow': [19],
    'elephant': [20],
    'bear': [21],
    'zebra': [22],
    'giraffe': [23],
    'animals': [15, 16, 17, 18, 19, 20, 21, 22, 23]
}

# COCO skeleton definition with 17 keypoints
# Maps keypoint indices to anatomical names and defines skeleton structure
coco_skeleton = {
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
    "skeleton": [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
        [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1],
        [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]}


class ViTPoseClassifier(nn.Module):
    """
    Activity classifier based on ViTPose pose estimation.

    This model uses a pre-trained ViTPose backbone to extract human pose keypoints,
    then extracts geometric features (keypoint positions, distances, angles) and
    classifies activities using fully connected layers.

    The backbone (ViTPose) is frozen during training, and only the classification
    head is trained on the extracted pose features.

    Args:
        cfg (dict): Configuration dictionary for ViTPose model (backbone and head config)
        num_classes (int): Number of activity classes to predict
        dropout_rate (float): Dropout probability for regularization. Default: 0.5
        pre_trained_backbone_path (str, optional): Path to pre-trained ViTPose weights
        fc_size (int): Size of the first fully connected layer. Default: 512
    """
    def __init__(self, cfg: dict, num_classes: int, dropout_rate: float = 0.5, pre_trained_backbone_path: str = None,
                 fc_size=512):
        super(ViTPoseClassifier, self).__init__()

        # Extract and initialize ViTPose backbone and keypoint detection head
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}

        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)

        # Load pre-trained weights for the backbone if available
        if pre_trained_backbone_path:
            checkpoint = torch.load(pre_trained_backbone_path, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])

        # Calculate input size based on extracted features:
        input_size = self.calculate_input_size()

        # Classification head: two fully connected layers with dropout
        self.fc1 = nn.Linear(input_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Standard scaler for feature normalization (fitted during first training epoch)
        self.scaler = StandardScaler()

        # Training state flags
        self.first_epoch = True
        self.stored_backbone_features = None

        # Freeze the backbone and keypoint head - only train classification layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.keypoint_head.parameters():
            param.requires_grad = False

    def calculate_input_size(self):
        """
        Calculate the total size of the feature vector.

        Returns:
            int: Total number of features (keypoint coords + skeleton metrics)
        """
        # 3 features per keypoint (x, y, confidence) + 2 per skeleton edge (distance, angle)
        return 3 * len(list(coco_skeleton['keypoints'].keys())) + 2 * len(coco_skeleton['skeleton'])

    def forward_features(self, x):
        """
        Extract features using the ViTPose backbone.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Feature maps from the backbone
        """
        return self.backbone(x)

    def forward(self, x, img_conf):
        """
        Forward pass: extract pose features and classify activity.

        Args:
            x (torch.Tensor): Input image tensor [batch_size, channels, height, width]
            img_conf (dict): Image configuration with keys:
                - 'org_w': Original image width
                - 'org_h': Original image height
                - 'x_y': Top-left corner coordinates
                - 'pad': Padding applied to image

        Returns:
            torch.Tensor: Class probabilities [batch_size, num_classes]
        """
        # Extract pose keypoints from the backbone
        # Note: Backbone features can be cached during training to save computation
        # This optimization is currently disabled (see condition below)
        backbone_features = None
        if self.stored_backbone_features is not None and self.training and False:
            # Use cached features (disabled by default as it doesn't improve accuracy)
            backbone_features = self.stored_backbone_features
        else:
            # Temporarily disable training mode to ensure consistent pose estimation
            switched = False
            if self.training:
                self.eval()
                switched = True

            # Extract pose heatmaps and convert to keypoints
            backbone_features = self.keypoint_head(self.backbone(x)).detach().cpu().numpy()

            if switched:
                self.train()
                # Cache features for potential reuse (disabled by default)
                self.stored_backbone_features = backbone_features

        # Post-process heatmaps to get keypoint coordinates
        keypoints = self.postprocess(backbone_features, img_conf['org_w'], img_conf['org_h'])

        # Extract geometric features from keypoints
        features = self.extract_features(keypoints, img_conf['x_y'], img_conf['pad'])

        # Normalize features using StandardScaler
        if self.training and self.first_epoch:
            # Fit scaler on first epoch
            features = self.scaler.fit_transform(features)
            self.first_epoch = False
        else:
            # Use fitted scaler for transformation
            features = self.scaler.transform(features)

        # Convert to tensor and pass through classification head
        x = torch.from_numpy(features).to('cuda')
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), -1)
        x = self.dropout(x)

        return x

    def postprocess(self, heatmaps, org_ws, org_hs):
        """
        Convert pose heatmaps to keypoint coordinates.

        Args:
            heatmaps (numpy.ndarray): Pose estimation heatmaps from the model
            org_ws (torch.Tensor): Original image widths for each sample
            org_hs (torch.Tensor): Original image heights for each sample

        Returns:
            numpy.ndarray: Keypoint coordinates with confidence scores
                           Shape: [batch_size, num_keypoints, 3] (x, y, confidence)
        """
        org_ws = org_ws.detach().cpu().numpy()
        org_hs = org_hs.detach().cpu().numpy()

        # Calculate image centers and scales for keypoint transformation
        center = []
        scale = []
        for org_w, org_h in zip(org_ws, org_hs):
            center.append([org_w // 2, org_h // 2])
            scale.append([org_w, org_h])

        # Convert heatmaps to keypoint coordinates with confidence scores
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array(center),
                                               scale=np.array(scale),
                                               unbiased=True, use_udp=True)
        # Return keypoints with flipped x,y order and confidence
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    def extract_features(self, keypoints_batch, x_y_coords, pads):
        """
        Extract geometric features from pose keypoints.

        Creates a feature vector for each sample containing:
        1. Raw keypoint coordinates (x, y) and confidence for each of 17 keypoints
        2. Pairwise distances between connected keypoints (skeleton edges)
        3. Angles between connected keypoints

        Args:
            keypoints_batch (numpy.ndarray): Keypoint data [batch_size, num_keypoints, 3]
            x_y_coords (torch.Tensor): Top-left corner coordinates for each image
            pads (torch.Tensor): Padding applied to each image

        Returns:
            numpy.ndarray: Feature vectors [batch_size, feature_dim]
        """
        x_y_coords = x_y_coords.detach().cpu().numpy()
        pads = pads.detach().cpu().numpy()

        extracted_features = []

        for body_keypoints, x_y, pad in zip(keypoints_batch, x_y_coords, pads):
            # Transform keypoints back to original image coordinates
            body_keypoints[:, :2] += x_y - pad

            feature_vector = []

            # Add keypoint positions and confidence scores
            for kp_index in coco_skeleton['keypoints'].keys():
                x = body_keypoints[kp_index][0]
                y = body_keypoints[kp_index][1]
                confidence = body_keypoints[kp_index][2]
                feature_vector.append(x)
                feature_vector.append(y)
                feature_vector.append(confidence)

            # Add skeletal features: distance and angle between connected keypoints
            for kp_index_0, kp_index_1 in coco_skeleton['skeleton']:
                v1 = body_keypoints[kp_index_0]
                v2 = body_keypoints[kp_index_1]

                # Euclidean distance between keypoints
                dist = math.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
                # Angle of the vector connecting the keypoints
                angle = math.atan2(v2[1] - v1[1], v2[0] - v1[0])

                feature_vector.append(dist)
                feature_vector.append(angle)

            extracted_features.append(feature_vector)

        return np.array(extracted_features, dtype=np.float32)
