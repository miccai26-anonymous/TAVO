#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

# =====================================================
# Config
# =====================================================
PROJECT_ROOT = Path(
    "/path/to/workspace/"
    "Tumor_Segmentation_Summer2025_XWDR/"
    "ANON_USER/EfficientVit"
)

SCORE_ROOT = PROJECT_ROOT / "data/splits_TCGA_mix_scores"
OUT_ROOT   = PROJECT_ROOT / "data/splits_TCGA_cma"


# =====================================================
# Utils
# =====================================================
def load_norm_scores():
    """
    Load normalized RDS / LESS scores (TCGA)
    """
    rds  = np.load(SCORE_ROOT / "rds_norm_dict.npy",  allow_pickle=True).item()
    less = np.load(SCORE_ROOT / "less_norm_dict.npy", allow_pickle=True).item()
    return rds, less


def build_subset(rds, less, w_rds, budget):
    """
    Weighted ranking and top-k selection
    """
    scores = {
        k: w_rds * rds[k] + (1.0 - w_rds) * less[k]
        for k in rds
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [k for k, _ in ranked[:budget]]
    return selected


def write_split(out_dir, subjects):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_subjects.txt", "w") as f:
        for sid in subjects:
            f.write(f"{sid}\n")


# =====================================================
# Main
# =====================================================
def main(args):
    budget_T = args.budget_T
    T        = args.T
    w_list   = args.w_list

    budget = budget_T * T

    print("=" * 60)
    print("🔧 Building TCGA grid splits (RDS + LESS)")
    print(f"budget_T = {budget_T}T  (={budget} subjects)")
    print(f"w_list   = {w_list}")
    print("=" * 60)

    rds, less = load_norm_scores()

    for w in w_list:
        w = float(w)
        assert 0.0 <= w <= 1.0

        tag = f"wR{w:.3f}_wL{1.0-w:.3f}"
        out_dir = OUT_ROOT / f"{budget_T}T" / tag

        subset = build_subset(
            rds,
            less,
            w_rds=w,
            budget=budget,
        )

        write_split(out_dir, subset)
        print(f"✅ {tag}: {len(subset)} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_T", type=int, required=True,
                        help="Multiplier of T (e.g. 1, 5, 10)")
    parser.add_argument("--T", type=int, default=50,
                        help="Target subject count T (default: 50)")
    parser.add_argument(
        "--w_list",
        type=float,
        nargs="+",
        required=True,
        help="list of w_rds values, e.g. 0 0.25 0.5 0.75 1",
    )

    args = parser.parse_args()
    main(args)
