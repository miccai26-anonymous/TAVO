#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression


# ============================================================
# GradMatch FULL ranking (OMP-style greedy)
# ============================================================
def gradmatch_full_ranking(src_vecs, tgt_vecs,
                           normalize=True,
                           max_rank=750,
                           eps=1e-12):

    Ns, D = src_vecs.shape
    K = min(max_rank, Ns)

    g_t = tgt_vecs.mean(axis=0).astype(np.float64)
    G = src_vecs.astype(np.float64)

    if normalize:
        G = G / (np.linalg.norm(G, axis=1, keepdims=True) + eps)
        g_t = g_t / (np.linalg.norm(g_t) + eps)

    selected = []
    residual = g_t.copy()
    prev_r2 = float(np.dot(residual, residual))

    print("🚀 Running GradMatch OMP ranking...")

    for step in range(K):

        corr = G @ residual

        if selected:
            corr[np.array(selected)] = -np.inf

        j = int(np.argmax(corr))
        selected.append(j)

        # NNLS refit
        A = G[selected]
        X = A.T
        y = g_t

        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X, y)
        w_sel = reg.coef_

        approx = X @ w_sel
        residual = y - approx

        r2 = float(np.dot(residual, residual))
        prev_r2 = r2

        if step % 100 == 0:
            print(f"Step {step}/{K} | residual norm² = {r2:.6f}")

    return selected


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max_rank", type=int, default=750)
    args = parser.parse_args()

    base_dir = "results/orient_embeddings_TCGA"
    out_dir  = "data/splits_TCGA_gradmatch"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Loading embeddings from: {base_dir}")

    src_vecs = np.load(os.path.join(base_dir, "src_case_vecs.npy"))
    tgt_vecs = np.load(os.path.join(base_dir, "tgt_case_vecs.npy"))

    with open(os.path.join(base_dir, "src_case_ids.txt")) as f:
        src_ids = [line.strip() for line in f]

    print(f"Source subjects: {len(src_ids)}")
    print(f"Target subjects: {tgt_vecs.shape[0]}")

    # ============================================================
    # 1️⃣ FULL RANKING (first 750)
    # ============================================================
    selected_order = gradmatch_full_ranking(
        src_vecs,
        tgt_vecs,
        normalize=args.normalize,
        max_rank=args.max_rank
    )

    # ============================================================
    # 2️⃣ Build full score dict
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
        os.path.join(out_dir, "gradmatch_score_dict.npy"),
        score_dict,
        allow_pickle=True
    )

    print("💾 Saved gradmatch_score_dict.npy")

    # ============================================================
    # 3️⃣ Generate budget subsets
    # ============================================================
    budgets_T = [1, 5, 10, 15]

    for k in budgets_T:

        budget = k * args.T

        subset_ids = [
            src_ids[i]
            for i in selected_order[:budget]
        ]

        subset_dir = os.path.join(out_dir, f"gradmatch_{k}T")
        os.makedirs(subset_dir, exist_ok=True)

        with open(os.path.join(subset_dir, "train_subjects.txt"), "w") as f:
            f.write("\n".join(subset_ids))

        print(f"✅ Saved gradmatch_{k}T ({budget})")

    print("\n🎉 TCGA GradMatch complete!")


if __name__ == "__main__":
    main()
