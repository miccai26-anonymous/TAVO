"""GradMatch selection using Orthogonal Matching Pursuit."""

from typing import Dict, List, Optional

import numpy as np

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("gradmatch")
class GradMatchSelection(SelectionMethod):
    """
    GradMatch with Orthogonal Matching Pursuit (OMP).

    Selects samples whose gradients best match the query set gradient
    using greedy orthogonal matching pursuit.
    """

    name = "gradmatch"
    requires_embeddings = False
    requires_gradients = True
    requires_query_set = True

    def __init__(self, seed: int = 42, omp: bool = True):
        super().__init__(seed)
        self.omp = omp

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
            raise ValueError("GradMatch requires pool gradients")
        if query_gradients is None:
            raise ValueError("GradMatch requires query gradients")

        # Target gradient (mean of query gradients)
        target = query_gradients.mean(axis=0)
        target = target / (np.linalg.norm(target) + 1e-8)

        pool_grads = gradients.copy()
        n = len(pool_grads)
        budget = min(budget, n)

        # Normalize pool gradients
        norms = np.linalg.norm(pool_grads, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        pool_normalized = pool_grads / norms

        selected: List[int] = []
        weights: Dict[int, float] = {}
        scores = np.zeros(n)

        residual = target.copy()

        for step in range(budget):
            # Find gradient with highest correlation to residual
            correlations = pool_normalized @ residual
            correlations[selected] = -np.inf

            idx = int(np.argmax(correlations))
            selected.append(idx)

            if self.omp:
                # OMP: recompute optimal weights for all selected
                A = pool_normalized[selected].T  # (D, k)
                weights_vec, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                for i, sel_idx in enumerate(selected):
                    weights[sel_idx] = float(max(weights_vec[i], 0))

                # Update residual
                residual = target - A @ weights_vec
            else:
                # Simple: just subtract projection
                proj = correlations[idx] * pool_normalized[idx]
                residual = residual - proj
                weights[idx] = float(max(correlations[idx], 0))

            # Score based on selection order and weight
            scores[idx] = (budget - step) / budget

        # Score unselected by correlation to target
        unselected_mask = scores == 0
        if unselected_mask.any():
            correlations = np.abs(pool_normalized @ target)
            correlations[~unselected_mask] = 0
            if correlations.max() > 0:
                scores[unselected_mask] = correlations[unselected_mask] / correlations.max() * (0.5 / budget)

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        selected_ids = [pool_ids[i] for i in selected]

        return SelectionResult(
            selected=selected_ids,
            scores=scores_dict,
            method=self.name + ("_omp" if self.omp else ""),
            budget=budget,
            metadata={
                "seed": self.seed,
                "omp": self.omp,
                "weights": {pool_ids[k]: v for k, v in weights.items()},
            },
        )
