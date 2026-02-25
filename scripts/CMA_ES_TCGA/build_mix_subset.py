#!/usr/bin/env python3
import os
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True, help="Number of subjects to select")
    ap.add_argument("--w", type=float, required=True, help="Weight for RDS (0~1)")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    w = float(args.w)
    assert 0.0 <= w <= 1.0, "w must be in [0, 1]"

    # --------------------------------------------------
    # Load normalized score dicts (TCGA)
    # --------------------------------------------------
    norm_dir = "data/splits_TCGA_mix_scores"

    rds_path = os.path.join(norm_dir, "rds_norm_dict.npy")
    less_path = os.path.join(norm_dir, "less_norm_dict.npy")

    assert os.path.exists(rds_path), f"Missing {rds_path}"
    assert os.path.exists(less_path), f"Missing {less_path}"

    rds = np.load(rds_path, allow_pickle=True).item()
    less = np.load(less_path, allow_pickle=True).item()

    # --------------------------------------------------
    # Sanity check: subject alignment
    # --------------------------------------------------
    keys = set(rds.keys())
    assert keys == set(less.keys()), "❌ Subject ID mismatch between RDS and LESS!"

    print(f"✔ Loaded {len(keys)} aligned subjects")

    # --------------------------------------------------
    # Mixed score: RDS + LESS
    # --------------------------------------------------
    mix = {
        sid: w * rds[sid] + (1.0 - w) * less[sid]
        for sid in keys
    }

    ranked = sorted(mix.items(), key=lambda x: x[1], reverse=True)
    selected = [sid for sid, _ in ranked[:args.budget]]

    # --------------------------------------------------
    # Save subset
    # --------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = os.path.join(args.out_dir, "train_subjects.txt")

    with open(out_txt, "w") as f:
        f.write("\n".join(selected))

    print(f"✅ Saved mixed subset → {out_txt}")
    print(f"   w = {w:.3f}, budget = {args.budget}")


if __name__ == "__main__":
    main()
