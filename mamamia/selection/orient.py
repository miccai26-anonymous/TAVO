"""Orient SMI (Submodular Mutual Information) selection via submodlib."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from submodlib.functions.facilityLocationMutualInformation import (
    FacilityLocationMutualInformationFunction,
)

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("orient")
class OrientSelection(SelectionMethod):
    """
    Orient: Facility Location Mutual Information based selection.

    Uses submodlib's FacilityLocationMutualInformationFunction to
    greedily maximize mutual information between selected pool samples
    and the query set.
    """

    name = "orient"
    requires_embeddings = True
    requires_gradients = False
    requires_query_set = True

    def __init__(self, seed: int = 42, eta: float = 1.0):
        super().__init__(seed)
        self.eta = eta

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
            raise ValueError("Orient requires pool embeddings")
        if query_embeddings is None:
            raise ValueError("Orient requires query embeddings")

        pool_np = embeddings.astype(np.float32)
        query_np = query_embeddings.astype(np.float32)
        n = len(pool_np)
        budget = min(budget, n)

        # Similarity matrices (clip negatives to 0, matching BraTS pipeline)
        K = np.maximum(cosine_similarity(pool_np, pool_np), 0).astype(np.float32)
        Q = np.maximum(cosine_similarity(pool_np, query_np), 0).astype(np.float32)

        obj = FacilityLocationMutualInformationFunction(
            n=n,
            num_queries=query_np.shape[0],
            data_sijs=K,
            query_sijs=Q,
            magnificationEta=self.eta,
        )

        result = obj.maximize(
            budget=budget,
            optimizer="LazyGreedy",
            stopIfNegativeGain=False,
            show_progress=False,
        )

        ordered_idx = []
        gains = []
        for idx, gain in result:
            ordered_idx.append(int(idx))
            gains.append(float(gain))

        # Rank-based scores: highest rank = highest score
        scores = np.zeros(n)
        L = len(ordered_idx)
        for rank, idx in enumerate(ordered_idx):
            scores[idx] = 1.0 - rank / max(L - 1, 1)

        # Score unselected by mean query similarity (scaled below selected)
        unselected_mask = scores == 0
        if unselected_mask.any():
            relevance = Q.mean(axis=1)
            rel_unseen = relevance[unselected_mask]
            if rel_unseen.max() > 0:
                scores[unselected_mask] = rel_unseen / rel_unseen.max() * (0.5 / budget)

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        selected_ids = [pool_ids[ordered_idx[i]] for i in range(min(budget, L))]

        return SelectionResult(
            selected=selected_ids,
            scores=scores_dict,
            method=self.name,
            budget=budget,
            metadata={"seed": self.seed, "eta": self.eta},
        )
