#!/usr/bin/env python3
import os
import argparse
import numpy as np


# ============================================================
# KCenter Greedy Ranking (Farthest-First Traversal)
# ============================================================
def kcenter_full_ranking(X, max_rank=750, normalize=True, eps=1e-12):

    N, D = X.shape
    K = min(max_rank, N)

    X = X.astype(np.float64)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    print(f"🔹 Running KCenter greedy (max_rank={K})...")

    selected = []

    # 1️⃣ random start
    first = np.random.randint(N)
    selected.append(first)

    # distance to selected set
    dists = np.linalg.norm(X - X[first], axis=1)

    for step in range(1, K):

        # pick farthest point
        idx = np.argmax(dists)
        selected.append(idx)

        # update distances
        new_dist = np.linalg.norm(X - X[idx], axis=1)
        dists = np.minimum(dists, new_dist)

        if step % 50 == 0:
            print(f"Step {step}/{K}")

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

    # 🔥 TCGA feature embedding path
    base_dir = "data/splits_TCGA_rds"
    out_dir  = "data/splits_TCGA_kcenter"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading TCGA feature embeddings from: {base_dir}")

    src_vecs = np.load(os.path.join(base_dir, "src_subject_vecs.npy"))

    with open(os.path.join(base_dir, "src_subject_ids.txt")) as f:
        src_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(src_ids)}")

    # ============================================================
    # 1️⃣ KCenter Ranking
    # ============================================================
    selected_order = kcenter_full_ranking(
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
        os.path.join(out_dir, "kcenter_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )

    print("💾 Saved kcenter_score_dict.npy")

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

        subset_dir = os.path.join(out_dir, f"kcenter_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved kcenter_{k}T ({budget})")

    print("\n🎉 TCGA KCenter full ranking + subsets complete!")


if __name__ == "__main__":
    main()
