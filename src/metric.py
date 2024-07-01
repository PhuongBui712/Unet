import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def pixel_accuracy(logits: Tensor, masks: Tensor):
    """
    Calculate the pixel accuracy of the predicted masks.

    Args:
        logits (Tensor): A tensor of shape (N, C, H, W) containing the logits for each class.
        masks (Tensor): A tensor of shape (N, H, W) containing the ground truth masks.

    Returns:
        float: The pixel accuracy of the predicted masks.
    """
    with torch.no_grad():
        prob = F.softmax(logits, dim=1)
        predicted_mask = torch.argmax(prob, dim=1)

        correct_pred = (predicted_mask == masks)
        accuracy = torch.sum(correct_pred).item() / correct_pred.numel()

    return accuracy


def mean_iou(logits: Tensor, masks: Tensor, num_classes: int):
    """
    Calculate the mean Intersection over Union (IoU) of the predictions.

    Args:
        logits (Tensor): A tensor of shape (N, C, H, W) containing the logits for each class.
        masks (Tensor): A tensor of shape (N, H, W) containing the ground truth masks.
        num_classes (int): The number of classes in the dataset.

    Returns:
        float: The mean IoU of the predicted masks.
    """
    with torch.no_grad():
        pred_masks = F.softmax(logits, dim=1)
        pred_masks = torch.argmax(pred_masks, dim=1)

        iou_per_class = []
        for cls in range(num_classes):
            pred_inds = (pred_masks == cls)
            target_inds = (masks == cls)

            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                iou_per_class.append(np.nan)
            else:
                iou_per_class.append((pred_inds & union).sum().item() / union)

        return np.nanmean(iou_per_class)
            