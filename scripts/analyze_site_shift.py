#!/usr/bin/env python3
import argparse
import numpy as np
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


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

    Kxx = k_rbf(A, A)
    Kyy = k_rbf(B, B)
    Kxy = k_rbf(A, B)
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def domain_clf_accuracy(X, y, seed=0):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y_enc):
        clf = LogisticRegression(max_iter=2000, n_jobs=1)
        clf.fit(X[tr], y_enc[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y_enc[te], pred))
    return float(np.mean(accs)), float(np.std(accs))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", type=str, required=True,
                    help="merged all_sites_epoch*.npz OR single-split npz")

    ap.add_argument("--split", type=str, default=None,
                    choices=[
                        "source_train",
                        "target_train",
                        "target_val",
                        "target_test",
                        "target_all",
                        "all_union",   # ✅ 新增：source + target(all)
                    ],
                    help="Which split to use (for merged npz)")

    ap.add_argument("--target_sites", type=str, default=None,
                    help="Comma-separated target sites (required unless split=target_all)")

    ap.add_argument("--min_n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    pack = np.load(args.npz, allow_pickle=True)

    Xt = None

    # ========================================================
    # Load data
    # ========================================================
    if args.split is not None:

        # --- 1) 旧：target_all = 用 source_train 做site列表 + target(all)做Xt（你之前那套） ---
        if args.split == "target_all":
            X = pack["source_train_vecs"].astype(np.float32)
            site_names = pack["source_train_sites"]

            Xt = np.vstack([
                pack["target_train_vecs"],
                pack["target_val_vecs"],
                pack["target_test_vecs"],
            ]).astype(np.float32)

            print("Loaded MERGED npz | split = target_all")

        # --- 2) ✅ 新：all_union = X包含所有site（source + TCGA），用于“任选site vs 其余所有site” ---
        elif args.split == "all_union":
            X = np.vstack([
                pack["source_train_vecs"],
                pack["target_train_vecs"],
                pack["target_val_vecs"],
                pack["target_test_vecs"],
            ]).astype(np.float32)

            site_names = np.concatenate([
                pack["source_train_sites"],
                pack["target_train_sites"],
                pack["target_val_sites"],
                pack["target_test_sites"],
            ]).astype(object)

            print("Loaded MERGED npz | split = all_union (source + target union)")

        # --- 3) 其它 split：直接取对应键 ---
        else:
            vec_key = f"{args.split}_vecs"
            site_key = f"{args.split}_sites"
            if vec_key not in pack:
                raise KeyError(f"{vec_key} not found in {args.npz}")

            X = pack[vec_key].astype(np.float32)
            site_names = pack[site_key]
            print(f"Loaded MERGED npz | split = {args.split}")

    else:
        # single-split npz
        X = pack["subj_vecs"].astype(np.float32)
        site_names = pack["site_names"]
        print("Loaded SINGLE-SPLIT npz")

    # normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    if Xt is not None:
        Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)

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

    # ========================================================
    # Target definition
    # ========================================================
    if Xt is None:
        if args.target_sites is None:
            raise ValueError("Must provide --target_sites unless split=target_all")

        target_sites = [s.strip() for s in args.target_sites.split(",") if s.strip()]
        for t in target_sites:
            if t not in site_to_idx:
                raise ValueError(f"Target site '{t}' not found (or < min_n / UNKNOWN)")

        Xt = np.vstack([X[site_to_idx[t]] for t in target_sites]).astype(np.float32)
        Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
        print(f"Target sites: {target_sites}")

    print(f"Target total N = {Xt.shape[0]}")
    print(f"Sites considered (min_n={args.min_n}): {len(site_to_idx)}")

    # ========================================================
    # Distance ranking
    # ========================================================
    rows = []
    for site, idxs in site_to_idx.items():
        Xs = X[idxs]
        d_cent = centroid_cosine_distance(Xs, Xt)
        mmd2 = mmd_rbf(Xs, Xt, seed=args.seed)
        rows.append((site, len(idxs), d_cent, mmd2))

    rows.sort(key=lambda x: (x[2], x[3]))

    print("\n=== Site → Target distance ranking ===")
    print(f"{'site':35s} {'N':>6s} {'centroid_cos_dist':>18s} {'MMD^2(RBF)':>12s}")
    for site, n, dcent, mmd2 in rows:
        print(f"{site:35s} {n:6d} {dcent:18.4f} {mmd2:12.4f}")

    # ========================================================
    # Domain separability
    # ========================================================
    all_X, all_sites = [], []
    for site, idxs in site_to_idx.items():
        all_X.append(X[idxs])
        all_sites.extend([site] * len(idxs))

    all_X = np.vstack(all_X)
    all_sites = np.array(all_sites, dtype=object)

    acc_mean, acc_std = domain_clf_accuracy(all_X, all_sites, seed=args.seed)
    print("\n=== Domain separability sanity check ===")
    print(f"LogReg site-classification acc (5-fold): {acc_mean:.3f} ± {acc_std:.3f}")
    print("Higher acc ⇒ stronger domain shift.")


if __name__ == "__main__":
    main()
