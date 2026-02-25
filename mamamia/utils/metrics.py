"""Metrics for segmentation evaluation."""

import numpy as np
import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Dice score for binary segmentation (2D slice).

    Args:
        pred: Predicted mask (H, W) or (B, H, W), values 0 or 1
        target: Ground truth mask, same shape as pred
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice score in [0, 1]
    """
    pred = (pred > 0).float()
    target = (target > 0).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return ((2.0 * intersection + smooth) / (union + smooth)).item()


def dice_score_3d(pred_volume: np.ndarray, gt_volume: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Compute 3D Dice score for a full volume.

    Args:
        pred_volume: Predicted 3D mask (D, H, W)
        gt_volume: Ground truth 3D mask (D, H, W)
        smooth: Smoothing factor

    Returns:
        3D Dice score in [0, 1]
    """
    pred_binary = (pred_volume > 0).astype(np.float32)
    gt_binary = (gt_volume > 0).astype(np.float32)

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection + smooth) / (union + smooth)
