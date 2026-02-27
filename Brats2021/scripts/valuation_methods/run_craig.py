#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from submodlib.functions.facilityLocation import FacilityLocationFunction


# ============================================================
# CRAIG Full Ranking (Source Coverage, marginal gain score)
# ============================================================
def craig_full_ranking(src_vecs, max_rank, normalize=True, eps=1e-12):

    Ns, D = src_vecs.shape
    max_rank = min(max_rank, Ns - 1)

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
            idx, gain = elem
        else:
            idx = elem
            gain = 0.0
        selected.append(idx)
        gains.append(float(gain))

    return selected, gains


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser("CRAIG for UPENN / IVYGAP / C5 / TCGA_LGG / TCGA_GBM")
    parser.add_argument("--target", required=True, choices=["UPENN", "IVYGAP", "C5", "TCGA_LGG", "TCGA_GBM"])
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--max_rank", type=int, default=750)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Embedding path (reuses ORIENT gradient embeddings)
    # --------------------------------------------------------
    embed_root = (
        "/path/to/workspace/"
        "ANON_USER/EfficientVit/results/orient_embeddings_"
        f"{args.target}"
    )

    out_root = (
        "/path/to/workspace/"
        "ANON_USER/EfficientVit/data/splits_"
        f"{args.target}_craig"
    )

    os.makedirs(out_root, exist_ok=True)

    print(f"\n📂 Loading embeddings from: {embed_root}")

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------
    src_vecs = np.load(os.path.join(embed_root, "src_case_vecs.npy"))

    with open(os.path.join(embed_root, "src_case_ids.txt")) as f:
        src_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(src_ids)}")

    # ============================================================
    # 1️⃣ CRAIG Full Ranking (marginal gain)
    # ============================================================
    selected_order, gains = craig_full_ranking(
        src_vecs,
        max_rank=args.max_rank,
        normalize=args.normalize
    )

    # ============================================================
    # 2️⃣ Build Score Dict (🔥 marginal gain-based)
    # ============================================================
    score = np.zeros(len(src_ids))

    for idx, gain in zip(selected_order, gains):
        score[idx] = float(gain)

    score_dict = {
        src_ids[i]: float(score[i])
        for i in range(len(src_ids))
    }

    score_path = os.path.join(out_root, "craig_score_dict.npy")
    np.save(score_path, score_dict, allow_pickle=True)

    print(f"💾 Saved craig_score_dict.npy → {score_path}")

    # Save greedy order (for debugging)
    ordered_ids = [src_ids[i] for i in selected_order]
    with open(os.path.join(out_root, "craig_sorted_ids.txt"), "w") as f:
        f.write("\n".join(ordered_ids))

    print("💾 Saved greedy order.")

    # ============================================================
    # 3️⃣ Generate Budget Subsets
    # ============================================================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:

        budget = k * args.T

        subset_ids = [
            src_ids[i]
            for i in selected_order[:budget]
        ]

        subset_dir = os.path.join(out_root, f"craig_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved craig_{k}T ({budget})")

    print("\n🎉 CRAIG completed (marginal gain score).")


if __name__ == "__main__":
    main()
