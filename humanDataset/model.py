import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset

# ==========================
# Dataset
# ==========================
class HumanSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask shape is (1,H,W)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
            mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask, self.image_paths[idx]

# ==========================
# Model Builder (DeepLabV3Plus, 3 Encoder)
# ==========================
def build_models():
    """
    Returns a dictionary of DeepLabV3Plus models with 3 different encoders.
    """
    encoders = ["resnet50", "resnet101", "mobilenet_v2"]
    models = {}
    for enc in encoders:
        models[enc] = smp.DeepLabV3Plus(
            encoder_name=enc,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder={'dropout': 0.3, 'aspp_dropout': 0.5}
        )
    return models

# ==========================
# Metrics
# ==========================
def calculate_metrics(pred, mask, threshold=0.4, use_logits=False):
    smooth = 1e-6

    if use_logits:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()

    TP = torch.sum(pred_binary * mask).item()
    FP = torch.sum(pred_binary * (1 - mask)).item()
    FN = torch.sum((1 - pred_binary) * mask).item()
    TN = torch.sum((1 - pred_binary) * (1 - mask)).item()

    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    iou = TP / (TP + FP + FN + smooth)
    specificity = TN / (TN + FP + smooth)
    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)

    return precision, recall, f1, iou, specificity, accuracy

# ==========================
# Data Loader Helper
# ==========================
def get_images_and_masks(dataset_path, split):
    images_dir = os.path.join(dataset_path, "images", split)
    masks_dir  = os.path.join(dataset_path, "masks", split)

    images = sorted(os.listdir(str(images_dir)))
    masks = sorted(os.listdir(str(masks_dir)))

    assert len(images) == len(masks), f"{split} image-mask count mismatch"

    image_paths = [os.path.join(str(images_dir), f) for f in images]
    mask_paths  = [os.path.join(str(masks_dir), f) for f in masks]

    return image_paths, mask_paths

# ==========================
# Loss Functions
# ==========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

class CombinedLoss(nn.Module):
    """
    Combined Dice + Focal BCE Loss
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=0.85, gamma=1.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalBCELoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal