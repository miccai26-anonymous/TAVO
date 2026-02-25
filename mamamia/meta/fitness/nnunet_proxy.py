"""
nnUNet proxy fitness evaluator for meta-optimization.

Uses PlainConvUNet (same architecture as real nnUNet 2d training) with
Dice + CE loss. Trains for N steps and returns validation Dice as fitness.
This gives a more relevant and stable fitness signal than the EfficientViT proxy.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...utils.data_loading import MamamiaSliceDataset


class SoftDiceLoss(nn.Module):
    """Soft Dice loss matching nnUNet's implementation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, C, H, W) logits
        # target: (B, H, W) integer labels
        pred_soft = torch.softmax(pred, dim=1)
        n_classes = pred_soft.shape[1]

        # One-hot encode target
        target_oh = torch.zeros_like(pred_soft)
        target_oh.scatter_(1, target.unsqueeze(1), 1)

        # Dice per class (skip background)
        dice_sum = 0.0
        for c in range(1, n_classes):
            p = pred_soft[:, c].flatten(1)
            t = target_oh[:, c].flatten(1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice_sum += (2.0 * inter + self.smooth) / (union + self.smooth)

        return 1.0 - dice_sum.mean() / max(1, n_classes - 1)


class DCandCELoss(nn.Module):
    """Dice + Cross-Entropy loss matching nnUNet."""

    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight_ce * self.ce(pred, target) + self.weight_dice * self.dice(pred, target)


def create_nnunet_2d_model(in_channels: int = 3, num_classes: int = 2):
    """
    Create a PlainConvUNet matching nnUNet's 2d config.

    Uses dynamic_network_architectures if available, otherwise falls back
    to a simple UNet with the same structure.
    """
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        model = PlainConvUNet(
            input_channels=in_channels,
            n_stages=7,
            features_per_stage=[32, 64, 128, 256, 512, 512, 512],
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3]] * 7,
            strides=[[1, 1]] + [[2, 2]] * 6,
            n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            num_classes=num_classes,
        )
        return model
    except ImportError:
        raise ImportError(
            "dynamic_network_architectures required. "
            "Install: pip install dynamic_network_architectures"
        )


def compute_val_dice(model, val_loader, device):
    """Compute validation Dice score (foreground class)."""
    model.eval()
    total_inter = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            pred = outputs.argmax(dim=1)
            # Foreground dice
            total_inter += ((pred == 1) & (masks == 1)).sum().item()
            total_union += (pred == 1).sum().item() + (masks == 1).sum().item()

    model.train()
    if total_union == 0:
        return 0.0
    return 2.0 * total_inter / total_union


class NNUNetProxyFitnessEvaluator:
    """
    Evaluates fitness by training nnUNet's PlainConvUNet for N steps.

    Returns validation Dice as fitness (higher = better).
    Much more stable than EfficientViT proxy because:
    1. Same architecture as downstream evaluation
    2. Dice + CE loss (matches nnUNet)
    3. More training steps (500 vs 100)
    """

    def __init__(
        self,
        data_dirs: List[Path],
        val_cases: List[str],
        batch_size: int = 12,
        lr: float = 1e-3,
        weight_decay: float = 3e-5,
        crop_size: int = 256,
        best_k: int = 4,
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.crop_size = crop_size
        self.best_k = best_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Val dataset (shared across all evaluations)
        self.val_dataset = MamamiaSliceDataset(
            data_dirs=self.data_dirs,
            case_ids=val_cases,
            augment=False,
            min_tumor_pixels=50,
            crop_size=crop_size,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        print(f"  Val: {len(self.val_dataset)} slices from {len(val_cases)} cases")

    def evaluate(self, selected_cases: List[str], n_steps: int) -> Dict:
        """
        Train nnUNet for n_steps and return fitness + detailed metrics.

        Returns dict with:
            fitness: float (median of best-k validation dices, higher = better)
            val_dices: list of validation dices at each checkpoint
            train_losses: list of training losses
            final_val_dice: last validation dice
        """
        # Create training dataset
        train_dataset = MamamiaSliceDataset(
            data_dirs=self.data_dirs,
            case_ids=selected_cases,
            augment=True,
            min_tumor_pixels=50,
            crop_size=self.crop_size,
        )
        if len(train_dataset) == 0:
            return {"fitness": float("-inf"), "val_dices": [], "train_losses": [], "error": "empty_dataset"}

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )

        # Create model
        model = create_nnunet_2d_model(in_channels=3, num_classes=2).to(self.device)
        criterion = DCandCELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Training loop
        model.train()
        train_iter = iter(train_loader)
        train_losses = []
        val_dices = []
        eval_every = max(1, n_steps // 10)  # 10 checkpoints

        for step in range(n_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer.step()

            train_losses.append(loss.item())

            # Evaluate periodically
            if (step + 1) % eval_every == 0:
                dice = compute_val_dice(model, self.val_loader, self.device)
                val_dices.append(dice)

        # Fitness: median of best-k validation dices
        if val_dices:
            sorted_dices = sorted(val_dices, reverse=True)
            best_k_dices = sorted_dices[: self.best_k]
            fitness = float(np.median(best_k_dices))
        else:
            fitness = float("-inf")

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()

        return {
            "fitness": fitness,
            "val_dices": val_dices,
            "train_losses": train_losses,
            "final_val_dice": val_dices[-1] if val_dices else None,
            "n_train_slices": len(train_dataset),
        }
