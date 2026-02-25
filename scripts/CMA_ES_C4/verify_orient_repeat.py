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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()

    base_dir = f"./data/splits_C4_orient_repeat{args.repeat:02d}_epoch3"
    score_path = os.path.join(base_dir, "orient_score_dict.npy")
    subset_5T_path = os.path.join(base_dir, "orient_5T", "train_subjects.txt")

    print("\n==============================")
    print(f"🔍 Verifying ORIENT for repeat{args.repeat:02d}")
    print(f"Path: {base_dir}")
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

    # ======================================================
    # Level 3: Subject ID alignment (with RDS as reference)
    # ======================================================
    print("\n[Level 3] Subject ID alignment with RDS")

    # we only check ID set consistency, not score values
    rds_path = f"./data/splits_C4_rds_repeat{args.repeat:02d}/rds_score_dict.npy"
    assert os.path.exists(rds_path), f"❌ Missing {rds_path}"

    rds_scores = np.load(rds_path, allow_pickle=True).item()

    orient_ids = set(orient_scores.keys())
    rds_ids = set(rds_scores.keys())

    missing = sorted(rds_ids - orient_ids)
    extra = sorted(orient_ids - rds_ids)

    print(f"  Missing in ORIENT : {len(missing)}")
    print(f"  Extra in ORIENT   : {len(extra)}")

    # ======================================================
    # Level 4: Reproduce orient_5T subset
    # ======================================================
    print("\n[Level 4] Reproduce orient_5T subset")

    assert os.path.exists(subset_5T_path), f"❌ Missing {subset_5T_path}"
    saved_subset = load_subset(subset_5T_path)

    # regenerate from score dict
    T = 50
    K = 5 * T

    regenerated = [
        sid for sid, _ in sorted_items if orient_scores[sid] > 0
    ][:K]

    match = set(saved_subset) == set(regenerated)
    print(f"  Match with saved orient_5T? {match}")

    print("\n✅ ORIENT verification completed successfully.")


if __name__ == "__main__":
    main()
