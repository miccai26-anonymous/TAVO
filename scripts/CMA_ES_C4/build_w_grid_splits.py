#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import argparse

# =====================================================
# Config (根据你的工程结构)
# =====================================================
PROJECT_ROOT = Path("/path/to/project")

SCORE_ROOT = PROJECT_ROOT / "data/splits_C4_mix_scores"
OUT_ROOT   = PROJECT_ROOT / "data/splits_C4_cma"

# =====================================================
# Utils
# =====================================================
def load_norm_scores(repeat_id):
    """
    Load normalized RDS / LESS scores
    """
    base = SCORE_ROOT / f"repeat{repeat_id:02d}"
    rds  = np.load(base / "rds_norm_dict.npy",  allow_pickle=True).item()
    less = np.load(base / "less_norm_dict.npy", allow_pickle=True).item()
    return rds, less


def build_subset(rds, less, w_rds, w_less, budget):
    """
    Weighted ranking and top-k selection
    """
    scores = {
        k: w_rds * rds[k] + w_less * less[k]
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
    repeat_id = args.repeat
    budget_T  = args.budget_T
    T         = args.T
    w_list    = args.w_list

    budget = budget_T * T

    print("=" * 60)
    print(f"🔧 Building grid splits")
    print(f"repeat   = {repeat_id:02d}")
    print(f"budget_T = {budget_T}T  (={budget} subjects)")
    print(f"w_list   = {w_list}")
    print("=" * 60)

    rds, less = load_norm_scores(repeat_id)

    for w in w_list:
        w_rds  = float(w)
        w_less = 1.0 - w_rds

        tag = f"wR{w_rds:.3f}_wL{w_less:.3f}"
        out_dir = (
            OUT_ROOT
            / f"repeat{repeat_id:02d}"
            / f"{budget_T}T"
            / tag
        )

        subset = build_subset(
            rds,
            less,
            w_rds=w_rds,
            w_less=w_less,
            budget=budget,
        )

        write_split(out_dir, subset)
        print(f"✅ {tag}: {len(subset)} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--budget_T", type=int, required=True)
    parser.add_argument("--T", type=int, default=10,
                        help="number of subjects per T (default: 10)")
    parser.add_argument(
        "--w_list",
        type=float,
        nargs="+",
        required=True,
        help="list of w_rds values, e.g. 0 0.25 0.5 0.75 1",
    )

    args = parser.parse_args()
    main(args)
