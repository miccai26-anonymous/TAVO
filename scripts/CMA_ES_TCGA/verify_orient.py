#!/usr/bin/env python3
import os
import argparse
import numpy as np


def load_score_dict(path):
    score_dict = np.load(path, allow_pickle=True).item()
    assert isinstance(score_dict, dict)
    return score_dict


def load_subset(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def verify_tcga_orient(base_dir, T=50):
    base = os.path.join(base_dir, "splits_TCGA_orient")
    score_path = os.path.join(base, "orient_score_dict.npy")
    subset_5T_path = os.path.join(base, "orient_5T", "train_subjects.txt")

    print("\n==============================")
    print("🔍 Verifying ORIENT for TCGA")
    print(f"Path: {base}")
    print("==============================")

    # ======================================================
    # Load
    # ======================================================
    assert os.path.exists(score_path), f"❌ Missing {score_path}"
    orient_scores = load_score_dict(score_path)

    print(f"✔ Loaded ORIENT dict with {len(orient_scores)} subjects")

    scores = np.array(list(orient_scores.values()), dtype=np.float32)

    # ======================================================
    # Level 1: Basic statistics
    # ======================================================
    print("\n[Level 1] Basic stats")
    print(f"  Min / Mean / Max : {scores.min():.6f} / {scores.mean():.6f} / {scores.max():.6f}")
    print(f"  Std              : {scores.std():.6f}")
    print(f"  Any NaN?          : {np.isnan(scores).any()}")
    print(f"  Any Inf?          : {np.isinf(scores).any()}")

    assert not np.isnan(scores).any(), "❌ NaN detected in ORIENT scores"
    assert not np.isinf(scores).any(), "❌ Inf detected in ORIENT scores"

    # ======================================================
    # Level 2: Ranking consistency (Top-10)
    # ======================================================
    print("\n[Level 2] Ranking consistency check (Top-10)")

    sorted_items = sorted(orient_scores.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_items[:10]

    monotonic = True
    prev_score = float("inf")

    for i, (sid, s) in enumerate(top10, 1):
        print(f"  #{i:02d} {sid}: {s:.6f}")
        if s > prev_score:
            monotonic = False
        prev_score = s

    print(f"  Monotonic decreasing? {monotonic}")
    assert monotonic, "❌ ORIENT top-10 scores are not monotonic"

    # ======================================================
    # Level 3: Subject ID alignment (with RDS)
    # ======================================================
    print("\n[Level 3] Subject ID alignment with RDS")

    rds_path = os.path.join(
        base_dir, "splits_TCGA_rds", "rds_score_dict.npy"
    )
    assert os.path.exists(rds_path), f"❌ Missing {rds_path}"

    rds_scores = np.load(rds_path, allow_pickle=True).item()

    orient_ids = set(orient_scores.keys())
    rds_ids = set(rds_scores.keys())

    missing = sorted(rds_ids - orient_ids)
    extra = sorted(orient_ids - rds_ids)

    print(f"  Missing in ORIENT : {len(missing)}")
    print(f"  Extra in ORIENT   : {len(extra)}")

    assert len(missing) == 0, "❌ Some RDS subjects missing in ORIENT"
    assert len(extra) == 0, "❌ Some ORIENT subjects not in RDS"

    # ======================================================
    # Level 4: Reproduce orient_5T subset
    # ======================================================
    if os.path.exists(subset_5T_path):
        print("\n[Level 4] Reproduce orient_5T subset")

        saved_subset = load_subset(subset_5T_path)

        K = 5 * T
        regenerated = [
            sid for sid, _ in sorted_items if orient_scores[sid] > 0
        ][:K]

        match = set(saved_subset) == set(regenerated)
        print(f"  Match with saved orient_5T? {match}")

        assert match, "❌ Recomputed ORIENT 5T does not match saved subset"
    else:
        print("\n[Level 4] orient_5T subset not found, skipped")

    print("\n✅ ORIENT verification completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data",
        help="Base directory containing splits_TCGA_orient"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=50,
        help="Target subject count T (default: 50)"
    )

    args = parser.parse_args()
    verify_tcga_orient(args.base_dir, T=args.T)
