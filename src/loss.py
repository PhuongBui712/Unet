"""
This file contains the implementation of the energy function for the U-Net model, 
as introduced in the original paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" 
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The energy function is used to measure 
the discrepancy between the predicted segmentation map and the ground truth segmentation map.

Note: This implementation is still in development and may not have the best precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class UNetLoss(nn.Module):
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
