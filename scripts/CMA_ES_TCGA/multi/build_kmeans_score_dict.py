#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans


# ============================================================
# KMeans Full Ranking (Feature Embedding Version)
# ============================================================
def kmeans_full_ranking(X, max_rank=750, normalize=True, eps=1e-12):

    N, D = X.shape
    K = min(max_rank, N)

    X = X.astype(np.float64)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    print(f"🔹 Running KMeans clustering (K={K})...")

    kmeans = KMeans(
        n_clusters=K,
        random_state=0,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    print("🔹 Selecting cluster representatives...")

    selected = []

    for c in range(K):

        cluster_idx = np.where(labels == c)[0]

        if len(cluster_idx) == 0:
            continue

        cluster_points = X[cluster_idx]
        center = centers[c]

        # pick point closest to centroid
        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_local = cluster_idx[np.argmin(dists)]

        selected.append(best_local)

    return selected


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--max_rank", type=int, default=750)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    # 🔥 TCGA feature embedding path (RDS embedding)
    base_dir = "data/splits_TCGA_rds"
    out_dir  = "data/splits_TCGA_kmeans"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading TCGA feature embeddings from: {base_dir}")

    src_vecs = np.load(os.path.join(base_dir, "src_subject_vecs.npy"))

    with open(os.path.join(base_dir, "src_subject_ids.txt")) as f:
        src_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(src_ids)}")

    # ============================================================
    # 1️⃣ KMeans Ranking
    # ============================================================
    selected_order = kmeans_full_ranking(
        src_vecs,
        max_rank=args.max_rank,
        normalize=args.normalize
    )

    # ============================================================
    # 2️⃣ Build Score Dict
    # ============================================================
    score = np.zeros(len(src_ids))
    N = len(selected_order)

    for rank, idx in enumerate(selected_order):
        score[idx] = 1.0 - rank / (N - 1)

    score_dict = {
        src_ids[i]: float(score[i])
        for i in range(len(src_ids))
    }

    np.save(
        os.path.join(out_dir, "kmeans_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )

    print("💾 Saved kmeans_score_dict.npy")

    # ============================================================
    # 3️⃣ Generate Budgets
    # ============================================================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:

        budget = k * args.T

        subset_ids = [
            src_ids[i]
            for i in selected_order[:budget]
        ]

        subset_dir = os.path.join(out_dir, f"kmeans_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved kmeans_{k}T ({budget})")

    print("\n🎉 TCGA KMeans full ranking + subsets complete!")


if __name__ == "__main__":
    main()
