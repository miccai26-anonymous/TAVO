"""Weighted linear combination of selection scores."""

from typing import Dict, List

import numpy as np

from ..base import ScoreCombiner
from ..registry import register_combiner


@register_combiner("weighted")
class WeightedCombiner(ScoreCombiner):
    """
    Linear weighted combination of normalized selection scores.

    Normalizes each method's scores to [0, 1], then computes
    weighted sum.
    """

    name = "weighted"

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize scores to [0, 1] before combining
        """
        self.normalize = normalize

    def combine(
        self,
        scores: Dict[str, Dict[str, float]],
        weights: np.ndarray,
        methods: List[str],
    ) -> Dict[str, float]:
        """Combine scores using weighted sum."""
        # Get all sample IDs
        all_samples = set()
        for method_scores in scores.values():
            all_samples.update(method_scores.keys())
        all_samples = sorted(all_samples)

        # Build score matrix
        n_samples = len(all_samples)
        n_methods = len(methods)
        score_matrix = np.zeros((n_samples, n_methods))

        sample_to_idx = {s: i for i, s in enumerate(all_samples)}

        for j, method in enumerate(methods):
            if method not in scores:
                continue
            method_scores = scores[method]
            for sample, score in method_scores.items():
                if sample in sample_to_idx:
                    score_matrix[sample_to_idx[sample], j] = score

        # Normalize each method's scores
        if self.normalize:
            for j in range(n_methods):
                col = score_matrix[:, j]
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    score_matrix[:, j] = (col - col_min) / (col_max - col_min)
                else:
                    score_matrix[:, j] = 0.5

        # Normalize weights
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_methods) / n_methods

        # Weighted combination
        combined = score_matrix @ weights

        return {all_samples[i]: float(combined[i]) for i in range(n_samples)}

    def select_top_k(
        self,
        scores: Dict[str, Dict[str, float]],
        weights: np.ndarray,
        methods: List[str],
        k: int,
    ) -> List[str]:
        """Combine scores and return top-k samples."""
        combined = self.combine(scores, weights, methods)
        sorted_samples = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
        return sorted_samples[:k]
