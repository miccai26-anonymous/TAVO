#!/usr/bin/env python3
import os
import argparse
import numpy as np
from collections import defaultdict


# ============================================================
# Rank Normalize (tie-aware)
# ============================================================
def rank_normalize(score_dict):

    items = list(score_dict.items())
    items.sort(key=lambda x: x[1], reverse=True)

    n = len(items)
    if n < 2:
        raise ValueError("Need at least 2 subjects.")

    ranks = {}
    value_to_indices = defaultdict(list)

    for idx, (sid, val) in enumerate(items):
        value_to_indices[val].append(idx)

    for val, indices in value_to_indices.items():
        avg_rank = sum(indices) / len(indices)
        for idx in indices:
            sid = items[idx][0]
            ranks[sid] = 1.0 - avg_rank / (n - 1)

    return ranks


# ============================================================
# MinMax Normalize
# ============================================================
def minmax_normalize(score_dict):

    vals = np.array(list(score_dict.values()), dtype=float)
    vmin = vals.min()
    vmax = vals.max()

    if vmax - vmin < 1e-12:
        raise ValueError("All scores identical.")

    return {
        k: (v - vmin) / (vmax - vmin)
        for k, v in score_dict.items()
    }


# ============================================================
# Load dict
# ============================================================
def load_dict(path, name):

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing {name}: {path}")

    data = np.load(path, allow_pickle=True).item()

    if not isinstance(data, dict):
        raise ValueError(f"{name} is not dict.")

    print(f"Loaded {name}: {len(data)} subjects")
    return data


# ============================================================
# Detect repeat folders
# ============================================================
def detect_repeats(base_dir, target, example_method):

    method_dir = os.path.join(
        base_dir,
        f"splits_{target}_{example_method}"
    )

    if not os.path.exists(method_dir):
        return []

    subdirs = [
        d for d in os.listdir(method_dir)
        if d.startswith("repeat")
        and os.path.isdir(os.path.join(method_dir, d))
    ]

    return sorted(subdirs)


# ============================================================
# Process one repeat (or single run)
# ============================================================
def process_one(args, repeat_id=None):

    print("\n===================================")
    print(f"Processing repeat = {repeat_id}")
    print("===================================\n")

    score_dicts = {}

    for m in args.methods:

        if repeat_id is None:
            path = os.path.join(
                args.base_dir,
                f"splits_{args.target}_{m}",
                f"{m}_score_dict.npy"
            )
        else:
            path = os.path.join(
                args.base_dir,
                f"splits_{args.target}_{m}",
                repeat_id,
                f"{m}_score_dict.npy"
            )

        score_dicts[m] = load_dict(path, m)

    # ---------- alignment check ----------
    key_sets = [set(d.keys()) for d in score_dicts.values()]
    base_keys = key_sets[0]

    for i, ks in enumerate(key_sets):
        if ks != base_keys:
            raise ValueError(f"❌ Subject mismatch in {args.methods[i]}")

    print(f"✅ All aligned. Subject count = {len(base_keys)}")

    # ---------- normalize ----------
    norm_dicts = {}

    for name, d in score_dicts.items():

        if args.mode == "rank":
            norm_dicts[name] = rank_normalize(d)
        else:
            norm_dicts[name] = minmax_normalize(d)

        print(f"Normalized {name}")

    # ---------- save ----------
    if repeat_id is None:
        out_dir = os.path.join(
            args.base_dir,
            f"splits_{args.target}_mix_scores_multi"
        )
    else:
        out_dir = os.path.join(
            args.base_dir,
            f"splits_{args.target}_mix_scores_multi",
            repeat_id
        )

    os.makedirs(out_dir, exist_ok=True)

    for name, d in norm_dicts.items():
        np.save(
            os.path.join(out_dir, f"{name}_norm_dict.npy"),
            d,
            allow_pickle=True
        )

    print(f"\n🎉 Saved → {out_dir}")


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser("Universal Mix Score Normalizer")

    parser.add_argument("--target", required=True)
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--mode", default="rank",
                        choices=["rank", "minmax"])
    parser.add_argument("--methods", nargs="+",
                        default=[
                            "rds",
                            "less",
                            "orient",
                            "gradmatch",
                            "craig",
                            "kmeans",
                            "kcenter",
                            "diversity"
                        ])

    args = parser.parse_args()

    print(f"\n🎯 Target = {args.target}")
    print(f"📊 Methods = {args.methods}")
    print(f"🔧 Mode = {args.mode}\n")

    # ---------- detect repeats ----------
    repeats = detect_repeats(
        args.base_dir,
        args.target,
        args.methods[0]
    )

    if repeats:
        print(f"Detected repeats = {repeats}\n")
        for r in repeats:
            process_one(args, r)
    else:
        print("No repeat folders detected. Single-run mode.\n")
        process_one(args, None)


if __name__ == "__main__":
    main()
