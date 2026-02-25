"""Simple embedding-based selection methods: K-Center, Herding, K-Means, Diversity."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .base import SelectionMethod, SelectionResult
from . import register_method


@register_method("kcenter")
class KCenterSelection(SelectionMethod):
    """K-Center (Farthest-First Traversal) selection."""

    name = "kcenter"
    requires_embeddings = True
    requires_query_set = True

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
            raise ValueError("K-Center requires embeddings")
        if query_embeddings is None:
            raise ValueError("K-Center requires query embeddings")

        pool_np = embeddings
        query_centroid = query_embeddings.mean(axis=0)

        n = len(pool_np)
        budget = min(budget, n)
        selected: List[int] = []
        scores = np.zeros(n)

        dists_to_query = np.linalg.norm(pool_np - query_centroid, axis=1)
        min_dist_to_selected = np.full(n, np.inf)

        for step in range(budget):
            if step == 0:
                idx = int(np.argmin(dists_to_query))
            else:
                min_dist_to_selected[selected] = -np.inf
                idx = int(np.argmax(min_dist_to_selected))

            selected.append(idx)
            scores[idx] = (budget - step) / budget
            new_dists = np.linalg.norm(pool_np - pool_np[idx], axis=1)
            min_dist_to_selected = np.minimum(min_dist_to_selected, new_dists)

        unselected_mask = scores == 0
        if unselected_mask.any():
            min_dist_to_selected[selected] = 0
            unselected_dists = min_dist_to_selected[unselected_mask]
            if unselected_dists.max() > 0:
                normalized = unselected_dists / unselected_dists.max() * (0.5 / budget)
                scores[unselected_mask] = normalized

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        selected_ids = [pool_ids[i] for i in selected]

        return SelectionResult(
            selected=selected_ids, scores=scores_dict, method=self.name,
            budget=budget, metadata={"seed": self.seed},
        )


@register_method("herding")
class HerdingSelection(SelectionMethod):
    """Herding selection - matches selected set mean to query mean."""

    name = "herding"
    requires_embeddings = True
    requires_query_set = True

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
        if embeddings is None or query_embeddings is None:
            raise ValueError("Herding requires embeddings and query embeddings")

        pool_np = embeddings
        query_mean = query_embeddings.mean(axis=0)

        n = len(pool_np)
        budget = min(budget, n)
        selected: List[int] = []
        scores = np.zeros(n)
        current_sum = np.zeros_like(query_mean)

        for step in range(budget):
            best_idx, best_dist = -1, np.inf
            for i in range(n):
                if i in selected:
                    continue
                new_mean = (current_sum + pool_np[i]) / (step + 1)
                dist = np.linalg.norm(new_mean - query_mean)
                if dist < best_dist:
                    best_dist, best_idx = dist, i

            selected.append(best_idx)
            current_sum += pool_np[best_idx]
            scores[best_idx] = (budget - step) / budget

        unselected_mask = scores == 0
        if unselected_mask.any():
            dists = np.linalg.norm(pool_np - query_mean, axis=1)
            dists[~unselected_mask] = np.inf
            if dists[unselected_mask].max() > 0:
                normalized = 1 - (dists[unselected_mask] / dists[unselected_mask].max())
                scores[unselected_mask] = normalized * (0.5 / budget)

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        return SelectionResult(
            selected=[pool_ids[i] for i in selected], scores=scores_dict,
            method=self.name, budget=budget, metadata={"seed": self.seed},
        )


@register_method("kmeans")
class KMeansSelection(SelectionMethod):
    """K-Means selection - cluster and select closest to centroids."""

    name = "kmeans"
    requires_embeddings = True

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
            raise ValueError("K-Means requires embeddings")

        pool_np = embeddings
        n = len(pool_np)
        budget = min(budget, n)

        kmeans = KMeans(n_clusters=budget, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(pool_np)
        centroids = kmeans.cluster_centers_

        selected: List[int] = []
        for k in range(budget):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) == 0:
                continue
            dists = np.linalg.norm(pool_np[cluster_indices] - centroids[k], axis=1)
            selected.append(int(cluster_indices[int(np.argmin(dists))]))

        dists_to_centroid = np.linalg.norm(pool_np - centroids[labels], axis=1)
        max_dist = dists_to_centroid.max()
        scores = 1 - (dists_to_centroid / max_dist) if max_dist > 0 else np.ones(n)

        for rank, idx in enumerate(selected):
            scores[idx] = 1.0 + (budget - rank) / budget
        scores = scores / scores.max()

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        return SelectionResult(
            selected=[pool_ids[i] for i in selected], scores=scores_dict,
            method=self.name, budget=budget, metadata={"seed": self.seed, "n_clusters": budget},
        )


@register_method("diversity")
class DiversitySelection(SelectionMethod):
    """Diversity selection - greedy DPP approximation."""

    name = "diversity"
    requires_embeddings = True

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
            raise ValueError("Diversity requires embeddings")

        pool_np = embeddings
        n = len(pool_np)
        budget = min(budget, n)

        norms = np.linalg.norm(pool_np, axis=1, keepdims=True)
        pool_normalized = pool_np / np.maximum(norms, 1e-8)

        selected: List[int] = []
        scores = np.zeros(n)

        idx = np.random.randint(n)
        selected.append(idx)
        scores[idx] = 1.0

        for step in range(1, budget):
            min_similarities = np.ones(n) * np.inf
            for sel_idx in selected:
                sims = np.abs(pool_normalized @ pool_normalized[sel_idx])
                min_similarities = np.minimum(min_similarities, sims)
            min_similarities[selected] = np.inf
            idx = int(np.argmin(min_similarities))
            selected.append(idx)
            scores[idx] = (budget - step) / budget

        unselected_mask = scores == 0
        if unselected_mask.any():
            min_sims = np.ones(n)
            for sel_idx in selected:
                sims = np.abs(pool_normalized @ pool_normalized[sel_idx])
                min_sims = np.minimum(min_sims, sims)
            diversity_scores = 1 - min_sims
            scores[unselected_mask] = diversity_scores[unselected_mask] * (0.5 / budget)

        scores_dict = {pool_ids[i]: float(scores[i]) for i in range(n)}
        return SelectionResult(
            selected=[pool_ids[i] for i in selected], scores=scores_dict,
            method=self.name, budget=budget, metadata={"seed": self.seed},
        )
