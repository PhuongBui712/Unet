"""
This file contains the implementation of the energy function for the U-Net model, 
as introduced in the original paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" 
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The energy function is used to measure 
the discrepancy between the predicted segmentation map and the ground truth segmentation map.

Note: This implementation is still in development and may not have the best precision.
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class UNetLoss(nn.Module):
    """
    This class implements the energy function for the U-Net model, as introduced in the original paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The energy function is used to measure the discrepancy between the predicted segmentation map and the ground truth segmentation map.

    Args:
        weight_map (torch.Tensor): A tensor of shape (N, 1, H, W) containing the weights for each pixel.
        num_classes (int): The number of classes in the dataset.

    Returns:
        torch.Tensor: The energy function value.
    """

    def __init__(self, weight_map, num_classes):
        super(UNetLoss, self).__init__()
        self.weight_map = weight_map
        self.num_classes = num_classes

    def forward(self, logits, target):
        # Apply softmax to the logits to get the probabilities
        probs = F.softmax(logits, dim=1)

        # Create one-hot encoding of the target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute the cross-entropy loss
        log_probs = torch.log(probs)
        loss = -self.weight_map * target_one_hot * log_probs
        loss = loss.sum(dim=1).mean()

        return loss
    

class DiceLoss(nn.Module):
    """
    DiceLoss class calculates the Dice coefficient loss, which is often used for 
    image segmentation tasks. This implementation supports both binary and 
    multiclass segmentation.

    Args:
        smooth (float): A smoothing constant to avoid division by zero errors. Default is 1e-10.

    Methods:
        forward(logits, masks):
            Computes the Dice loss between the predicted logits and the ground truth masks.

            Args:
                logits (Tensor): A tensor of shape (N, C, H, W) containing the predicted logits for each class.
                masks (Tensor): A tensor of shape (N, H, W) containing the ground truth masks.

            Returns:
                Tensor: The calculated Dice loss.
    """
    def __init__(self, smooth: float = 1e-10):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, masks: Tensor):
        # calculate probability for both logits and masks
        probs = F.softmax(logits, dim=1)

        one_hot_masks = F.one_hot(masks, num_classes=probs.shape[1])
        one_hot_masks = one_hot_masks.permute(0, 3, 1, 2).float()

        # flatten for element-wise operations
        probs = probs.view(probs.shape[0], probs.shape[1], -1)
        one_hot_masks = one_hot_masks.view(one_hot_masks.shape[0], one_hot_masks.shape[1], -1)
        # compute loss
        intersection = torch.sum(probs * one_hot_masks, dim=2)
        total = probs.sum(dim=2) + one_hot_masks.sum(dim=2)

        dice_coef = 2 * intersection / total
        avg_class_dice_coef = dice_coef.mean(dim=1)
        loss = 1 - avg_class_dice_coef.mean() # mean for batch

        return loss        

