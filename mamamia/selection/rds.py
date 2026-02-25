"""RDS+ (Representative Data Selection) with round-robin query matching."""

from typing import Dict, List, Optional

import numpy as np

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("rds")
class RDSSelection(SelectionMethod):
    """
    RDS+: Round-robin representative data selection.

    Each query sample takes turns picking the most similar
    unselected pool sample, ensuring balanced coverage across
    all query samples.
    """

    name = "rds"
    requires_embeddings = True
    requires_gradients = False
    requires_query_set = True

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
        if embeddings is None:
            raise ValueError("RDS+ requires pool embeddings")
        if query_embeddings is None:
            raise ValueError("RDS+ requires query embeddings")

        pool_np = embeddings
        n = len(pool_np)
        budget = min(budget, n)

        # Normalize embeddings
        pool_norms = np.linalg.norm(pool_np, axis=1, keepdims=True)
        pool_norms = np.maximum(pool_norms, 1e-8)
        pool_normalized = pool_np / pool_norms

        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_norms = np.maximum(query_norms, 1e-8)
        query_normalized = query_embeddings / query_norms

        # Cosine similarity: (num_queries, num_pool)
        similarity = query_normalized @ pool_normalized.T
        num_queries = len(query_normalized)

        # Round-robin selection: each query takes turns picking
        selected: List[int] = []
        used = np.zeros(n, dtype=bool)

        while len(selected) < budget and not used.all():
            progress = False
            for q_idx in range(num_queries):
                if len(selected) >= budget:
                    break
                q_scores = similarity[q_idx].copy()
                q_scores[used] = -np.inf
                best_idx = int(np.argmax(q_scores))
                if used[best_idx]:
                    continue
                used[best_idx] = True
                selected.append(best_idx)
                progress = True
            if not progress:
                break

        # Scores: selection order for selected, max similarity for all
        scores = np.zeros(n)
        for step, idx in enumerate(selected):
            scores[idx] = (budget - step) / budget

        # Score unselected by max similarity to any query
        max_sim = similarity.max(axis=0)  # (num_pool,)
        unselected_mask = scores == 0
        if unselected_mask.any() and max_sim[unselected_mask].max() > 0:
            scores[unselected_mask] = (
                max_sim[unselected_mask] / max_sim[unselected_mask].max() * (0.5 / budget)
            )

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        selected_ids = [pool_ids[i] for i in selected]

        return SelectionResult(
            selected=selected_ids,
            scores=scores_dict,
            method=self.name,
            budget=budget,
            metadata={"seed": self.seed},
        )
