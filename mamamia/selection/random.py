"""Random selection method."""

from typing import Dict, List, Optional

import numpy as np

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("random")
class RandomSelection(SelectionMethod):
    """Uniform random selection from pool."""

    name = "random"
    requires_embeddings = False
    requires_gradients = False
    requires_query_set = False

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
        """Randomly select samples from pool."""
        n = len(pool_ids)
        budget = min(budget, n)

        # Random permutation
        indices = np.random.permutation(n)
        selected_indices = indices[:budget]

        # Scores: selected get (budget - rank) / budget, unselected get 0
        scores: Dict[str, float] = {}
        for rank, idx in enumerate(selected_indices):
            scores[pool_ids[idx]] = (budget - rank) / budget

        # Unselected get small random scores
        for idx in indices[budget:]:
            scores[pool_ids[idx]] = np.random.uniform(0, 0.5 / budget)

        selected = [pool_ids[i] for i in selected_indices]

        return SelectionResult(
            selected=selected,
            scores=scores,
            method=self.name,
            budget=budget,
            metadata={"seed": self.seed},
        )
