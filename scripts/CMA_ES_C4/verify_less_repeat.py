#!/usr/bin/env python3
import os
import argparse
import numpy as np


def verify_repeat(base_dir, repeat_id, T=50):
    """
    Verify LESS score dict for one repeat.
    """
    base = os.path.join(base_dir, f"splits_C4_less_repeat{repeat_id:02d}")
    print(f"\n==============================")
    print(f"🔍 Verifying LESS for repeat{repeat_id:02d}")
    print(f"Path: {base}")
    print(f"==============================")

    # -----------------------------
    # Load files
    # -----------------------------
    dict_path = os.path.join(base, "less_score_dict.npy")
    subset_5T_path = os.path.join(base, "less_5T", "train_subjects.txt")

    assert os.path.exists(dict_path), f"❌ Missing {dict_path}"

    less = np.load(dict_path, allow_pickle=True).item()
    print(f"✔ Loaded LESS dict with {len(less)} subjects")

    # -----------------------------
    # Level 1: Basic sanity check
    # -----------------------------
    print("\n[Level 1] Basic stats")

    vals = np.array(list(less.values()))

    print(f"  Min / Mean / Max : {vals.min():.6f} / {vals.mean():.6f} / {vals.max():.6f}")
    print(f"  Std              : {vals.std():.6f}")
    print(f"  Any NaN?          : {np.isnan(vals).any()}")
    print(f"  Any Inf?          : {np.isinf(vals).any()}")

    assert not np.isnan(vals).any(), "❌ NaN detected in LESS scores"
    assert not np.isinf(vals).any(), "❌ Inf detected in LESS scores"

    # -----------------------------
    # Level 2: Ranking consistency
    # -----------------------------
    print("\n[Level 2] Ranking consistency check (Top-10)")

    sorted_ids = sorted(less, key=less.get, reverse=True)
    top10_scores = [less[sid] for sid in sorted_ids[:10]]

    monotonic = all(
        top10_scores[i] >= top10_scores[i + 1]
        for i in range(len(top10_scores) - 1)
    )

    for i, sid in enumerate(sorted_ids[:10]):
        print(f"  #{i+1:02d} {sid}: {less[sid]:.6f}")

    print(f"  Monotonic decreasing? {monotonic}")
    assert monotonic, "❌ LESS top-10 scores are not monotonic"

    # -----------------------------
    # Level 3: Subject ID alignment with RDS
    # -----------------------------
    print("\n[Level 3] Subject ID alignment with RDS")

    rds_path = os.path.join(
        base_dir, f"splits_C4_rds_repeat{repeat_id:02d}", "rds_score_dict.npy"
    )
    assert os.path.exists(rds_path), f"❌ Missing RDS dict at {rds_path}"

    rds = np.load(rds_path, allow_pickle=True).item()

    missing_in_less = set(rds) - set(less)
    extra_in_less = set(less) - set(rds)

    print(f"  Missing in LESS : {len(missing_in_less)}")
    print(f"  Extra in LESS   : {len(extra_in_less)}")

    assert len(missing_in_less) == 0, "❌ Some RDS subjects missing in LESS"
    assert len(extra_in_less) == 0, "❌ Some LESS subjects not in RDS"

    # -----------------------------
    # Level 4: Reproduce LESS 5T subset
    # -----------------------------
    if os.path.exists(subset_5T_path):
        print("\n[Level 4] Reproduce less_5T subset")

        with open(subset_5T_path) as f:
            original_5T = set(line.strip() for line in f)

        K = 5 * T
        recomputed = set(
            sorted(less, key=less.get, reverse=True)[:K]
        )

        match = original_5T == recomputed
        print(f"  Match with saved less_5T? {match}")

        if not match:
            print(f"  Original size : {len(original_5T)}")
            print(f"  Recomputed    : {len(recomputed)}")

        assert match, "❌ Recomputed LESS 5T does not match saved subset"
    else:
        print("\n[Level 4] less_5T subset not found, skipped")

    print("\n✅ LESS verification completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data",
        help="Base directory containing splits_C4_less_repeatXX"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=True,
        help="Repeat ID (e.g., 1 for repeat01)"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=50,
        help="Target subject count T (default: 50)"
    )

    args = parser.parse_args()
    verify_repeat(args.base_dir, args.repeat, T=args.T)
