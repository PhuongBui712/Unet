import torch
from torch.optim import Optimizer

import numpy as np
from scipy.ndimage import distance_transform_edt

        
def get_lr(optimier: Optimizer):
    """
    Get the current learning rate from the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer used in training.
    
    Returns:
        float: The current learning rate.
    """
    for param_group in optimier.param_groups:
        return param_group['lr']


def compute_weight_map(labels, w0=10, sigma=5):
    labels = labels.numpy()
    wc = np.zeros_like(labels, dtype=np.float32)
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_pixels = labels.size

    for label, count in zip(unique_labels, counts):
        wc[labels == label] = total_pixels / count

    distance = np.zeros_like(labels, dtype=np.float32)
    for label in unique_labels:
        if label == 0:
            continue
        binary_mask = labels == label
        distance += distance_transform_edt(~binary_mask)

    d1 = distance_transform_edt(labels == 1)
    d2 = distance_transform_edt(labels == 2)

    weight_map = wc + w0 * np.exp(- (d1 + d2) ** 2 / (2 * sigma ** 2))
    
    return torch.tensor(weight_map, dtype=torch.float32)
