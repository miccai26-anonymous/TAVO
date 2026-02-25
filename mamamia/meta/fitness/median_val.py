"""Training-based fitness evaluator using median of best validation losses."""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import FitnessEvaluator
from ..registry import register_fitness


@register_fitness("median_val")
class MedianValFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator using median of best K validation losses.

    fitness = -median(best_k_val_losses)

    Lower validation loss = higher fitness = better selection.
    Uses median for robustness against outliers.
    """

    name = "median_val"

    def __init__(
        self,
        data_dirs: List[Path],
        val_cases: List[str],
        model_factory,
        device: torch.device = None,
        n_steps: int = 100,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        eval_every: int = 10,
        best_k: int = 5,
    ):
        """
        Args:
            data_dirs: Directories containing preprocessed .npy files
            val_cases: Validation case IDs (fixed across all evaluations)
            model_factory: Callable that returns a fresh model instance
            device: Torch device
            n_steps: Number of training steps per evaluation
            batch_size: Batch size for training
            learning_rate: Learning rate
            eval_every: Evaluate on val set every N steps
            best_k: Number of best validation losses to take median of
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.val_cases = val_cases
        self.model_factory = model_factory
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eval_every = eval_every
        self.best_k = best_k

        # Pre-load validation dataset (shared across all evaluations)
        from ...utils.data_loading import MamamiaSliceDataset
        self.val_dataset = MamamiaSliceDataset(
            data_dirs=self.data_dirs,
            case_ids=val_cases,
            augment=False,
            min_tumor_pixels=50,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

    def evaluate(
        self,
        selected_cases: List[str],
        **kwargs
    ) -> float:
        """
        Run short training and compute median of best K validation losses.

        Args:
            selected_cases: Case IDs selected for training

        Returns:
            Fitness value (higher is better, so we negate the median loss)
        """
        from ...utils.data_loading import MamamiaSliceDataset

        # Create training dataset from selected cases
        train_dataset = MamamiaSliceDataset(
            data_dirs=self.data_dirs,
            case_ids=selected_cases,
            augment=True,
            min_tumor_pixels=50,
        )

        if len(train_dataset) == 0:
            return float("-inf")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        # Fresh model for this evaluation
        model = self.model_factory().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        # Training loop
        val_losses = []

        model.train()
        train_iter = iter(train_loader)

        for step in range(self.n_steps):
            # Get batch (cycle if needed)
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
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Evaluate on validation set periodically
            if (step + 1) % self.eval_every == 0 or step == 0:
                val_loss = self._compute_val_loss(model, criterion)
                val_losses.append(val_loss)

        # Get best K validation losses and compute median
        val_losses_sorted = sorted(val_losses)
        best_k_losses = val_losses_sorted[:self.best_k]

        if len(best_k_losses) == 0:
            fitness = float("-inf")
        else:
            # Negative because lower loss = better = higher fitness
            fitness = -np.median(best_k_losses)

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

        return float(fitness)

    def _compute_val_loss(self, model: nn.Module, criterion: nn.Module) -> float:
        """Compute average validation loss."""
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, masks)
                total_loss += loss.item() * images.shape[0]
                count += images.shape[0]

        model.train()
        return total_loss / count if count > 0 else float("inf")
