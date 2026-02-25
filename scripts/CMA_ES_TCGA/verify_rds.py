#!/usr/bin/env python3
import os
import argparse
import numpy as np


def verify_tcga(base_dir, T=50):
    """
    Verify RDS score dict for TCGA (no repeat).
    """
    base = os.path.join(base_dir, "splits_TCGA_rds")
    print(f"\n==============================")
    print(f"🔍 Verifying RDS for TCGA")
    print(f"Path: {base}")
    print(f"==============================")

    # -----------------------------
    # Load files
    # -----------------------------
    dict_path = os.path.join(base, "rds_score_dict.npy")
    sorted_ids_path = os.path.join(base, "rds_sorted_ids.txt")
    subset_5T_path = os.path.join(base, "rds_5T", "train_subjects.txt")

    assert os.path.exists(dict_path), f"Missing {dict_path}"
    assert os.path.exists(sorted_ids_path), f"Missing {sorted_ids_path}"

    rds = np.load(dict_path, allow_pickle=True).item()

    with open(sorted_ids_path) as f:
        sorted_ids = [line.strip() for line in f]

    print(f"✔ Loaded RDS dict with {len(rds)} subjects")

    # -----------------------------
    # Level 1: Basic sanity check
    # -----------------------------
    vals = np.array(list(rds.values()))

    print("\n[Level 1] Basic stats")
    print(f"  Min / Mean / Max : {vals.min():.4f} / {vals.mean():.4f} / {vals.max():.4f}")
    print(f"  Std              : {vals.std():.4f}")
    print(f"  Any NaN?          : {np.isnan(vals).any()}")
    print(f"  Any Inf?          : {np.isinf(vals).any()}")

    # -----------------------------
    # Level 2: Ranking consistency
    # -----------------------------
    print("\n[Level 2] Ranking consistency check (Top-10)")

    top10_scores = [rds[sid] for sid in sorted_ids[:10]]
    monotonic = all(
        top10_scores[i] >= top10_scores[i + 1]
        for i in range(len(top10_scores) - 1)
    )

    for i, sid in enumerate(sorted_ids[:10]):
        print(f"  #{i+1:02d} {sid}: {rds[sid]:.6f}")

    print(f"  Monotonic decreasing? {monotonic}")

    # -----------------------------
    # Level 3: Subject ID alignment
    # -----------------------------
    print("\n[Level 3] Subject ID alignment check")

    missing_ids = [sid for sid in sorted_ids if sid not in rds]
    extra_ids = [sid for sid in rds if sid not in sorted_ids]

    print(f"  Missing in dict : {len(missing_ids)}")
    print(f"  Extra in dict   : {len(extra_ids)}")

    assert len(missing_ids) == 0, "❌ Some sorted IDs not in rds dict"
    assert len(extra_ids) == 0, "❌ Some dict IDs not in sorted list"

    # -----------------------------
    # Level 4: Reproduce rds_5T subset
    # -----------------------------
    if os.path.exists(subset_5T_path):
        print("\n[Level 4] Reproduce rds_5T subset")

        with open(subset_5T_path) as f:
            original_5T = set(line.strip() for line in f)

        K = 5 * T
        recomputed = set(
            sorted(rds, key=rds.get, reverse=True)[:K]
        )

        match = original_5T == recomputed
        print(f"  Match with saved rds_5T? {match}")

        if not match:
            print(f"  Original size : {len(original_5T)}")
            print(f"  Recomputed    : {len(recomputed)}")
    else:
        print("\n[Level 4] rds_5T subset not found, skipped")

    print("\n✅ Verification completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data",
        help="Base directory containing splits_TCGA_rds"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=50,
        help="Target subject count T (default: 50)"
    )

    args = parser.parse_args()
    verify_tcga(args.base_dir, T=args.T)
