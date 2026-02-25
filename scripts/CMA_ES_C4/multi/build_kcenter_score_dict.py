#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances


# ============================================================
# K-Center FULL ranking (Farthest-First Traversal)
# ============================================================
def kcenter_full_ranking(X, max_rank=750, normalize=True):

    N, D = X.shape
    K = min(max_rank, N)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    print(f"🔹 Running K-Center (max_rank={K})...")

    # precompute distance matrix
    dist = pairwise_distances(X, metric="euclidean")

    selected = []
    min_dist = np.full(N, np.inf)

    # 1️⃣ 初始化：选 norm 最大的点
    norms = np.linalg.norm(X, axis=1)
    first = int(np.argmax(norms))
    selected.append(first)

    min_dist = np.minimum(min_dist, dist[:, first])

    # 2️⃣ Farthest-First Greedy
    for step in range(1, K):
        j = int(np.argmax(min_dist))
        selected.append(j)

        min_dist = np.minimum(min_dist, dist[:, j])

        if step % 100 == 0:
            print(f"Step {step}/{K}")

    return selected


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--max_rank", type=int, default=750)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    r = args.repeat

    base_dir = f"data/splits_C4_rds_repeat{r:02d}"
    out_dir  = f"data/splits_C4_kcenter_repeat{r:02d}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading feature embeddings from: {base_dir}")

    # load subject-level feature embedding (RDS已经算好的)
    X = np.load(os.path.join(base_dir, "src_subject_vecs.npy"))

    with open(os.path.join(base_dir, "src_subject_ids.txt")) as f:
        subject_ids = [line.strip() for line in f]

    print(f"Total subjects: {len(subject_ids)}")

    # ============================================================
    # 1️⃣ FULL ranking
    # ============================================================
    selected_order = kcenter_full_ranking(
        X,
        max_rank=args.max_rank,
        normalize=args.normalize
    )

    # ============================================================
    # 2️⃣ Build score dict
    # ============================================================
    score = np.zeros(len(subject_ids))
    N = len(selected_order)

    for rank, idx in enumerate(selected_order):
        score[idx] = 1.0 - rank / (N - 1)

    score_dict = {
        subject_ids[i]: float(score[i])
        for i in range(len(subject_ids))
    }

    np.save(
        os.path.join(out_dir, "kcenter_full_rank_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )

    print("💾 Saved kcenter_full_rank_score_dict.npy")

    # ============================================================
    # 3️⃣ Generate subsets
    # ============================================================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:

        budget = k * args.T

        subset_ids = [
            subject_ids[i]
            for i in selected_order[:budget]
        ]

        subset_dir = os.path.join(out_dir, f"kcenter_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved kcenter_{k}T ({budget})")

    print("\n🎉 K-Center full ranking + subsets complete!")


if __name__ == "__main__":
    main()
