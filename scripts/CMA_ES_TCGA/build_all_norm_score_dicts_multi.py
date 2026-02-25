#!/usr/bin/env python3
import os
import argparse
import numpy as np


# ============================================================
# Rank Normalize
# ============================================================
def rank_normalize(score_dict):
    items = list(score_dict.items())
    items.sort(key=lambda x: x[1], reverse=True)

    n = len(items)
    out = {}

    for rank, (sid, _) in enumerate(items):
        out[sid] = 1.0 - rank / (n - 1)

    return out


# ============================================================
# Load score dict safely
# ============================================================
def load_dict(path, name):

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing {name}: {path}")

    data = np.load(path, allow_pickle=True).item()

    print(f"Loaded {name}: {len(data)} subjects")

    return data


# ============================================================
# Main
# ============================================================
def main():

    # -----------------------------
    # TCGA paths (no repeat)
    # -----------------------------
    paths = {

        "rds":
            "data/splits_TCGA_rds/rds_score_dict.npy",

        "less":
            "data/splits_TCGA_less/less_score_dict.npy",

        "orient":
            "data/splits_TCGA_orient/orient_score_dict.npy",

        "gradmatch":
            "data/splits_TCGA_gradmatch/gradmatch_score_dict.npy",

        "craig":
            "data/splits_TCGA_craig/craig_score_dict.npy",

        "kmeans":
            "data/splits_TCGA_kmeans/kmeans_score_dict.npy",

        "kcenter":
            "data/splits_TCGA_kcenter/kcenter_score_dict.npy",

        "diversity":
            "data/splits_TCGA_diversity/diversity_score_dict.npy",
    }

    # -----------------------------
    # Load all
    # -----------------------------
    score_dicts = {}

    print("\n📂 Loading TCGA score dicts...\n")

    for name, path in paths.items():
        score_dicts[name] = load_dict(path, name)

    # -----------------------------
    # Alignment check
    # -----------------------------
    print("\n🔎 Checking subject alignment...")

    key_sets = [set(d.keys()) for d in score_dicts.values()]
    base_keys = key_sets[0]

    for i, ks in enumerate(key_sets):
        if ks != base_keys:
            raise ValueError(f"❌ Subject mismatch in method {list(paths.keys())[i]}")

    print(f"✅ All methods aligned. Subject count = {len(base_keys)}")

    # -----------------------------
    # Rank Normalize
    # -----------------------------
    norm_dicts = {}

    print("\n📊 Rank-normalizing...")

    for name, d in score_dicts.items():
        norm_dicts[name] = rank_normalize(d)
        print(f"Normalized {name}")

    # -----------------------------
    # Save
    # -----------------------------
    out_dir = "data/splits_TCGA_mix_scores_multi"
    os.makedirs(out_dir, exist_ok=True)

    for name, d in norm_dicts.items():
        np.save(
            os.path.join(out_dir, f"{name}_norm_dict.npy"),
            d,
            allow_pickle=True
        )

    print(f"\n🎉 Saved 8D normalized dicts → {out_dir}")


if __name__ == "__main__":
    main()
