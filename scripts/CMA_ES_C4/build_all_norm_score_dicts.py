#!/usr/bin/env python3
import os
import argparse
import numpy as np

def rank_normalize(score_dict):
    items = list(score_dict.items())
    items.sort(key=lambda x: x[1], reverse=True)  # high = good
    n = len(items)
    out = {}
    for rank, (sid, _) in enumerate(items):
        out[sid] = 1.0 - rank / (n - 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    args = ap.parse_args()
    r = args.repeat

    rds_path = f"data/splits_C4_rds_repeat{r:02d}/rds_score_dict.npy"
    less_path = f"data/splits_C4_less_repeat{r:02d}/less_score_dict.npy"
    orient_path = f"data/splits_C4_orient_repeat{r:02d}_epoch3/orient_score_dict.npy"

    rds = np.load(rds_path, allow_pickle=True).item()
    less = np.load(less_path, allow_pickle=True).item()
    orient = np.load(orient_path, allow_pickle=True).item()

    # sanity check: subject alignment
    keys = set(rds.keys())
    assert keys == set(less.keys()) == set(orient.keys()), "❌ Subject ID mismatch!"

    rds_norm = rank_normalize(rds)
    less_norm = rank_normalize(less)
    orient_norm = rank_normalize(orient)

    out_dir = f"data/splits_C4_mix_scores/repeat{r:02d}"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "rds_norm_dict.npy"), rds_norm, allow_pickle=True)
    np.save(os.path.join(out_dir, "less_norm_dict.npy"), less_norm, allow_pickle=True)
    np.save(os.path.join(out_dir, "orient_norm_dict.npy"), orient_norm, allow_pickle=True)

    print(f"✅ Saved normalized score dicts to {out_dir}")

if __name__ == "__main__":
    main()
