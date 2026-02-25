#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from submodlib.functions.facilityLocation import FacilityLocationFunction


# ============================================================
# CRAIG Full Ranking (Official gradient version)
# ============================================================
def craig_full_ranking(src_vecs, max_rank, normalize=True, eps=1e-12):

    Ns, D = src_vecs.shape
    max_rank = min(max_rank, Ns)

    X = src_vecs.astype(np.float64)

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    print("🔹 Computing cosine similarity matrix...")
    sim = cosine_similarity(X)
    sim = np.maximum(sim, 0.0).astype(np.float64)

    print("🔹 Building FacilityLocationFunction...")

    fl = FacilityLocationFunction(
        n=Ns,
        mode="dense",
        sijs=sim,
        separate_rep=False
    )

    print(f"🚀 Running LazyGreedy selection (max_rank={max_rank})")

    result = fl.maximize(
        budget=max_rank,
        optimizer="LazyGreedy",
        stopIfNegativeGain=False,
        show_progress=True
    )

    selected = []
    gains = []

    for elem in result:
        if isinstance(elem, tuple):
            selected.append(elem[0])
            gains.append(elem[1])
        else:
            selected.append(elem)
            gains.append(None)

    return selected, gains


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--max_rank", type=int, default=750)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    # 🔥 TCGA gradient embedding path
    base_dir = "results/orient_embeddings_TCGA"
    out_dir  = "data/splits_TCGA_craig"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading TCGA gradient embeddings from: {base_dir}")

    src_vecs = np.load(os.path.join(base_dir, "src_case_vecs.npy"))

    with open(os.path.join(base_dir, "src_case_ids.txt")) as f:
        src_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(src_ids)}")

    # ============================================================
    # 1️⃣ CRAIG Full Ranking
    # ============================================================
    selected_order, gains = craig_full_ranking(
        src_vecs,
        max_rank=args.max_rank,
        normalize=args.normalize
    )

    # ============================================================
    # 2️⃣ Build Full Score Dict (monotonic ranking)
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
        os.path.join(out_dir, "craig_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )

    print("💾 Saved craig_score_dict.npy")

    # ============================================================
    # 3️⃣ Generate all budgets automatically
    # ============================================================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:

        budget = k * args.T

        subset_ids = [
            src_ids[i]
            for i in selected_order[:budget]
        ]

        subset_dir = os.path.join(out_dir, f"craig_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved craig_{k}T ({budget})")

    print("\n🎉 TCGA Official CRAIG full ranking + subsets complete!")


if __name__ == "__main__":
    main()
