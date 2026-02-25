"""LESS (Low-rank Efficient Subset Selection) method."""

from typing import Dict, List, Optional

import numpy as np

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("less")
class LESSSelection(SelectionMethod):
    """
    LESS: Low-rank gradient approximation for efficient selection.

    Uses SVD to find low-rank approximation of gradient matrix,
    then selects samples that best span the gradient space.
    """

    name = "less"
    requires_embeddings = False
    requires_gradients = True
    requires_query_set = True

    def __init__(self, seed: int = 42, rank: int = 50):
        super().__init__(seed)
        self.rank = rank

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
            raise ValueError("LESS requires pool gradients")
        if query_gradients is None:
            raise ValueError("LESS requires query gradients")

        pool_grads = gradients
        n = len(pool_grads)
        budget = min(budget, n)
        rank = min(self.rank, n, pool_grads.shape[1])

        # Low-rank approximation of pool gradients
        U, S, Vt = np.linalg.svd(pool_grads, full_matrices=False)
        U_k = U[:, :rank]  # (n, rank)
        S_k = S[:rank]
        Vt_k = Vt[:rank, :]  # (rank, D)

        # Project query gradient to low-rank space
        target = query_gradients.mean(axis=0)
        target_proj = Vt_k @ target  # (rank,)

        # Weighted representation in low-rank space
        pool_proj = U_k * S_k  # (n, rank)

        # Greedy selection to match target
        selected: List[int] = []
        scores = np.zeros(n)
        residual = target_proj.copy()

        for step in range(budget):
            # Find sample with highest contribution to residual
            contributions = pool_proj @ residual
            contributions[selected] = -np.inf

            idx = int(np.argmax(contributions))
            selected.append(idx)
            scores[idx] = (budget - step) / budget

            # Update residual (orthogonal projection)
            proj = pool_proj[idx]
            proj_norm = np.linalg.norm(proj)
            if proj_norm > 1e-8:
                residual = residual - (proj @ residual) / (proj_norm ** 2) * proj

        # Score unselected
        unselected_mask = scores == 0
        if unselected_mask.any():
            contributions = np.abs(pool_proj @ target_proj)
            contributions[~unselected_mask] = 0
            if contributions.max() > 0:
                scores[unselected_mask] = contributions[unselected_mask] / contributions.max() * (0.5 / budget)

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        selected_ids = [pool_ids[i] for i in selected]

        return SelectionResult(
            selected=selected_ids,
            scores=scores_dict,
            method=self.name,
            budget=budget,
            metadata={"seed": self.seed, "rank": rank},
        )
