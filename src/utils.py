import torch
from torch.optim import Optimizer

import wandb
import numpy as np
from typing import Optional
from dotenv import load_dotenv
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


def get_gpu_count() -> int:
    """
    Get the number of available CUDA GPUs.
    
    Returns:
        int: The number of available CUDA GPUs.
    """
    return torch.cuda.device_count()


def setup_wandb(run_name: str,
                project: str,
                api_key: Optional[str] = None,
                epochs: Optional[int] = 10,
                batch: Optional[int] = 16) -> None:
    """
    Setup wandb for logging training progress.
    
    Args:
        run_name (str): The name of the wandb run.
        project (str): The name of the wandb project.
        api_key (Optional[str], optional): The wandb API key. Defaults to None.
        epochs (Optional[int], optional): The number of epochs for training. Defaults to 10.
        batch (Optional[int], optional): The batch size for training. Defaults to 16.
    """
    wandb_api_key = api_key
    if wandb_api_key:
        load_dotenv()

    wandb.login(key=wandb_api_key)

    wandb.init(
        name='[Unet] BCSS segmentation',
        config={
            'epoch': epochs,
            'batch_size': batch
        },
    )


def compute_weight_map(labels: torch.Tensor, w0: float = 10.0, sigma: float = 5.0):
    """
    Compute a weight map for the given labels.

    This function calculates a weight map that can be used to balance the 
    importance of different regions in an image segmentation task. The weight 
    map is computed based on the class frequencies and the distance to the 
    nearest boundary.

    Args:
        labels (torch.Tensor): A tensor containing the labels for each pixel 
            in the image. The labels should be integers representing different 
            classes.
        w0 (float, optional): A constant that controls the magnitude of the 
            weight map. Default is 10.
        sigma (float, optional): A constant that controls the spread of the 
            exponential function used to compute the weight map. Default is 5.

    Returns:
        torch.Tensor: A tensor containing the computed weight map.
    """
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
