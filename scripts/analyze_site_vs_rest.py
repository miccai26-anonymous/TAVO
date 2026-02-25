#!/usr/bin/env python3
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Distance metrics
# ============================================================

def centroid_cosine_distance(A, B):
    ma = A.mean(axis=0, keepdims=True)
    mb = B.mean(axis=0, keepdims=True)
    ma = ma / (np.linalg.norm(ma, axis=1, keepdims=True) + 1e-8)
    mb = mb / (np.linalg.norm(mb, axis=1, keepdims=True) + 1e-8)
    return float(1.0 - (ma @ mb.T)[0, 0])


def mmd_rbf(A, B, gamma=None, max_n=1000, seed=0):
    rng = np.random.RandomState(seed)

    if A.shape[0] > max_n:
        A = A[rng.choice(A.shape[0], max_n, replace=False)]
    if B.shape[0] > max_n:
        B = B[rng.choice(B.shape[0], max_n, replace=False)]

    if gamma is None:
        X = np.vstack([A, B])
        idx = rng.choice(X.shape[0], min(500, X.shape[0]), replace=False)
        Xs = X[idx]
        D = 1 - cosine_similarity(Xs, Xs)
        med = np.median(D[D > 0])
        gamma = 1.0 / (med**2 + 1e-8)

    def k_rbf(X, Y):
        sim = cosine_similarity(X, Y)
        dist = 1.0 - sim
        return np.exp(-gamma * (dist**2))

    return float(
        k_rbf(A, A).mean()
        + k_rbf(B, B).mean()
        - 2.0 * k_rbf(A, B).mean()
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True,
                    help="results/site_features/all_sites_epoch*.npz")

    ap.add_argument("--min_n", type=int, default=10,
                    help="Ignore sites with < min_n subjects")

    ap.add_argument("--topk", type=int, default=3,
                    help="Number of target sites to select")

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    pack = np.load(args.npz, allow_pickle=True)

    # ========================================================
    # 🔥 Build ALL-UNION feature matrix
    # ========================================================

    X_all = []
    site_all = []

    for split in [
        "source_train",
        "target_train",
        "target_val",
        "target_test",
    ]:
        vec_key = f"{split}_vecs"
        site_key = f"{split}_sites"

        if vec_key in pack:
            X_all.append(pack[vec_key])
            site_all.append(pack[site_key])

    X = np.vstack(X_all).astype(np.float32)
    site_names = np.concatenate(site_all)

    # normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # ========================================================
    # Group by site
    # ========================================================

    site_to_idx = defaultdict(list)
    for i, s in enumerate(site_names):
        site_to_idx[str(s)].append(i)

    site_to_idx = {
        k: v for k, v in site_to_idx.items()
        if len(v) >= args.min_n and k != "UNKNOWN"
    }

    sites = sorted(site_to_idx.keys())
    print(f"Sites considered (min_n={args.min_n}): {len(sites)}")

    # ========================================================
    # Site vs Rest
    # ========================================================

    rows = []

    for site in sites:
        idx_self = site_to_idx[site]
        idx_rest = [
            i for s, idxs in site_to_idx.items()
            if s != site
            for i in idxs
        ]

        A = X[idx_self]
        B = X[idx_rest]

        d_cent = centroid_cosine_distance(A, B)
        d_mmd = mmd_rbf(A, B, seed=args.seed)

        rows.append((site, len(idx_self), d_cent, d_mmd))

    rows.sort(key=lambda x: (x[2], x[3]), reverse=True)

    # ========================================================
    # Output
    # ========================================================

    print("\n=== Site vs Rest domain shift ranking ===")
    print(f"{'rank':>4s} {'site':35s} {'N':>6s} {'centroid_dist':>16s} {'MMD^2':>10s}")

    for i, (site, n, dc, dm) in enumerate(rows):
        print(f"{i+1:4d} {site:35s} {n:6d} {dc:16.4f} {dm:10.4f}")

    print("\n=== Recommended TARGET site set ===")
    for i in range(min(args.topk, len(rows))):
        site, n, _, _ = rows[i]
        print(f"  - {site} (N={n})")


if __name__ == "__main__":
    main()
