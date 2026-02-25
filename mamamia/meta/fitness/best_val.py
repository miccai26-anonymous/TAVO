"""Training-based fitness evaluator using best (minimum) validation loss."""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import FitnessEvaluator
from ..registry import register_fitness


@register_fitness("best_val")
class BestValFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator using the single best (minimum) validation loss.

    fitness = -min(val_losses)

    Simpler than median - just takes the best checkpoint.
    """

    name = "best_val"

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
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.val_cases = val_cases
        self.model_factory = model_factory
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eval_every = eval_every

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

    def evaluate(self, selected_cases: List[str], **kwargs) -> float:
        from ...utils.data_loading import MamamiaSliceDataset

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

        model = self.model_factory().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        val_losses = []
        model.train()
        train_iter = iter(train_loader)

        for step in range(self.n_steps):
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

            if (step + 1) % self.eval_every == 0 or step == 0:
                val_loss = self._compute_val_loss(model, criterion)
                val_losses.append(val_loss)

        # Best (minimum) validation loss, negated so higher = better
        fitness = -min(val_losses) if val_losses else float("-inf")

        del model, optimizer
        torch.cuda.empty_cache()

        return float(fitness)

    def _compute_val_loss(self, model: nn.Module, criterion: nn.Module) -> float:
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
