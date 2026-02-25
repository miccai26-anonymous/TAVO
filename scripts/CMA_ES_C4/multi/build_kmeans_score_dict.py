#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans


def kmeans_feature_ranking(X, max_rank=750, normalize=True):
    """
    X: (N, D) feature embedding
    Returns:
        full_order (length N)
    """

    N, D = X.shape
    K = min(max_rank, N)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    print(f"🔹 Running KMeans (K={K}) on feature embeddings...")

    kmeans = KMeans(
        n_clusters=K,
        random_state=0,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # cluster -> indices
    cluster_dict = {}
    for i, lab in enumerate(labels):
        cluster_dict.setdefault(lab, []).append(i)

    # ---- find representative per cluster ----
    reps = []
    rep_cluster_size = []

    for c, idxs in cluster_dict.items():
        cluster_points = X[idxs]
        center = centers[c]

        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_local = idxs[np.argmin(dists)]

        reps.append(best_local)
        rep_cluster_size.append(len(idxs))

    # ---- sort reps by cluster size descending ----
    reps_sorted = [
        x for _, x in sorted(
            zip(rep_cluster_size, reps),
            reverse=True
        )
    ]

    # ---- full ranking ----
    selected_set = set(reps_sorted)
    remaining = []

    for idx in range(N):
        if idx not in selected_set:
            remaining.append(idx)

    full_order = reps_sorted + remaining

    return full_order


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max_rank", type=int, default=750)
    args = parser.parse_args()

    r = args.repeat

    base_dir = f"data/splits_C4_rds_repeat{r:02d}"
    out_dir  = f"data/splits_C4_kmeans_repeat{r:02d}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading feature embeddings from: {base_dir}")

    X = np.load(os.path.join(base_dir, "src_subject_vecs.npy"))
    with open(os.path.join(base_dir, "src_subject_ids.txt")) as f:
        subject_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(subject_ids)}")

    # ===============================
    # 1️⃣ Full ranking
    # ===============================
    full_order = kmeans_feature_ranking(
        X,
        max_rank=args.max_rank,
        normalize=args.normalize
    )

    # ===============================
    # 2️⃣ Build full score dict
    # ===============================
    N = len(full_order)
    score = np.zeros(N)

    for rank, idx in enumerate(full_order):
        score[idx] = 1.0 - rank / (N - 1)

    score_dict = {
        subject_ids[i]: float(score[i])
        for i in range(N)
    }

    np.save(
        os.path.join(out_dir, "kmeans_feature_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )

    print("💾 Saved kmeans_feature_score_dict.npy")

    # ===============================
    # 3️⃣ Generate subsets
    # ===============================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:
        budget = k * args.T

        subset_ids = [
            subject_ids[i]
            for i in full_order[:budget]
        ]

        subset_dir = os.path.join(out_dir, f"kmeans_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved kmeans_{k}T ({budget})")

    print("\n🎉 KMeans(feature) ranking + subsets complete!")


if __name__ == "__main__":
    main()
