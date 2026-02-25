"""CRAIG proxy using embeddings instead of gradients.

This is an embedding-based approximation of CRAIG, using softmax-weighted
coverage scores as a proxy for facility location optimization.

For actual gradient-based CRAIG, see craig.py
"""

from typing import List, Optional

import numpy as np

from .base import SelectionMethod, SelectionResult
from . import register_method


def softmax(x, axis=None, temperature=1.0):
    """Compute softmax with temperature scaling."""
    x_scaled = x / temperature
    x_max = x_scaled.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


@register_method("craig_proxy")
class CRAIGProxySelection(SelectionMethod):
    """
    Embedding-based proxy for CRAIG (Coresets for Accelerating Incremental Gradient).

    Uses softmax-weighted coverage of query samples to approximate
    facility location scores.
    """

    name = "craig_proxy"
    requires_embeddings = True
    requires_gradients = False  # Uses embeddings as proxy
    requires_query_set = True

    def __init__(self, seed: int = 42, temperature: float = 5.0):
        """
        Args:
            seed: Random seed
            temperature: Temperature for softmax (higher = sharper coverage)
        """
        super().__init__(seed)
        self.temperature = temperature

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
            raise ValueError("CRAIG proxy requires pool embeddings")
        if query_embeddings is None:
            raise ValueError("CRAIG proxy requires query embeddings")

        n = len(pool_ids)
        budget = min(budget, n)

        # Normalize embeddings
        pool_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        pool_norms = np.maximum(pool_norms, 1e-8)
        pool_normalized = embeddings / pool_norms

        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_norms = np.maximum(query_norms, 1e-8)
        query_normalized = query_embeddings / query_norms

        # Cosine similarities: (N_pool, N_query)
        similarities = pool_normalized @ query_normalized.T

        # Softmax over pool dimension (for each query, how much each pool sample covers it)
        # Higher temperature = sharper coverage (more like max)
        coverage = softmax(similarities, axis=0, temperature=1.0/self.temperature)

        # Sum coverage across queries (total "facility location" score)
        scores = coverage.sum(axis=1)

        # Normalize scores to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores_normalized = (scores - s_min) / (s_max - s_min)
        else:
            scores_normalized = np.ones(n) * 0.5

        # Select top-k
        indices = np.argsort(scores_normalized)[::-1][:budget]
        selected_ids = [pool_ids[i] for i in indices]

        # Build scores dict
        scores_dict = {pool_ids[i]: float(scores_normalized[i]) for i in range(n)}

        return SelectionResult(
            selected=selected_ids,
            scores=scores_dict,
            method=self.name,
            budget=budget,
            metadata={"seed": self.seed, "proxy": True, "temperature": self.temperature},
        )
