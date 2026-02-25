#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Greedy Max-Diversity Ranking (Feature-based)
# ============================================================
def diversity_full_ranking(X, max_rank=750, normalize=True):
    N, D = X.shape
    K = min(int(max_rank), N)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    print(f"🔹 Running Diversity greedy ranking (max_rank={K})...")

    # cosine similarity -> distance
    sim = cosine_similarity(X)
    dist = 1.0 - sim

    selected = []

    # init: pick the most "spread" point (max avg distance)
    mean_dist = dist.mean(axis=1)
    first = int(np.argmax(mean_dist))
    selected.append(first)

    # min distance to selected set
    min_dist = dist[first].copy()

    for _ in tqdm(range(1, K), desc="Diversity ranking"):
        j = int(np.argmax(min_dist))
        selected.append(j)
        min_dist = np.minimum(min_dist, dist[j])

    return selected


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--max_rank", type=int, default=750)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    r = args.repeat

    # ✅ Correct feature-embedding source
    base_dir = f"data/splits_C4_rds_repeat{r:02d}"
    out_dir  = f"data/splits_C4_diversity_repeat{r:02d}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading feature embeddings from: {base_dir}")

    X_path = os.path.join(base_dir, "src_subject_vecs.npy")
    ids_path = os.path.join(base_dir, "src_subject_ids.txt")

    X = np.load(X_path)
    with open(ids_path) as f:
        ids = [line.strip() for line in f]

    assert X.shape[0] == len(ids), "❌ mismatch: src_subject_vecs vs src_subject_ids"
    print(f"Source subjects: {len(ids)}")

    # 1) ranking
    order = diversity_full_ranking(
        X,
        max_rank=args.max_rank,
        normalize=args.normalize
    )

    # 2) full rank score dict (monotonic)
    score = np.zeros(len(ids), dtype=np.float64)
    N = len(order)
    for rank, idx in enumerate(order):
        score[idx] = 1.0 - rank / max(1, (N - 1))

    score_dict = {ids[i]: float(score[i]) for i in range(len(ids))}

    np.save(
        os.path.join(out_dir, "diversity_full_rank_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )
    print("💾 Saved diversity_full_rank_score_dict.npy")

    # 3) subsets
    budgets_T = [1, 5, 10, 15]
    for k in budgets_T:
        budget = k * args.T
        subset_ids = [ids[i] for i in order[:budget]]

        subset_dir = os.path.join(out_dir, f"diversity_{k}T")
        os.makedirs(subset_dir, exist_ok=True)
        out_txt = os.path.join(subset_dir, "train_subjects.txt")
        with open(out_txt, "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved diversity_{k}T ({budget}) → {out_txt}")

    print("\n🎉 Diversity ranking + all subsets complete!")


if __name__ == "__main__":
    main()
