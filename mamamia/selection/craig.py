"""CRAIG (Coresets for Accelerating Incremental Gradient descent) selection."""

from typing import List, Optional

import numpy as np

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("craig")
class CRAIGSelection(SelectionMethod):
    """
    CRAIG: Coreset selection via facility location.

    Selects a coreset that approximates the full gradient by
    solving a facility location problem greedily.
    """

    name = "craig"
    requires_embeddings = False
    requires_gradients = True
    requires_query_set = False

    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def select(
        self,
        pool_ids: List[str],
        budget: int,
        embeddings: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None,
        query_embeddings: Optional[np.ndarray] = None,
        query_gradients: Optional[np.ndarray] = None,
        **kwargs
    ) -> SelectionResult:
        if gradients is None:
            raise ValueError("CRAIG requires gradients")

        pool_grads = gradients
        n = len(pool_grads)
        budget = min(budget, n)

        # Normalize gradients
        norms = np.linalg.norm(pool_grads, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        pool_normalized = pool_grads / norms

        # Compute similarity matrix (batched for memory efficiency)
        batch_size = 1000
        similarity = np.zeros((n, n), dtype=np.float32)
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            similarity[i:end_i] = pool_normalized[i:end_i] @ pool_normalized.T

        # Greedy facility location
        selected: List[int] = []
        scores = np.zeros(n)
        max_sim_to_selected = np.full(n, -np.inf)

        for step in range(budget):
            gains = np.zeros(n)
            gains[selected] = -np.inf

            for i in range(n):
                if i in selected:
                    continue
                new_coverage = np.maximum(similarity[:, i], max_sim_to_selected)
                gains[i] = new_coverage.sum() - max_sim_to_selected.sum()

            idx = int(np.argmax(gains))
            selected.append(idx)
            scores[idx] = (budget - step) / budget
            max_sim_to_selected = np.maximum(max_sim_to_selected, similarity[:, idx])

        # Score unselected
        unselected_mask = scores == 0
        if unselected_mask.any():
            coverage_scores = max_sim_to_selected.copy()
            coverage_scores[~unselected_mask] = 0
            if coverage_scores.max() > 0:
                scores[unselected_mask] = coverage_scores[unselected_mask] / coverage_scores.max() * (0.5 / budget)

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        selected_ids = [pool_ids[i] for i in selected]

        return SelectionResult(
            selected=selected_ids,
            scores=scores_dict,
            method=self.name,
            budget=budget,
            metadata={"seed": self.seed},
        )
