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

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()

    r = args.repeat

    # -----------------------------
    # Define all paths
    # -----------------------------
    paths = {
        "rds":
            f"data/splits_C4_rds_repeat{r:02d}/rds_score_dict.npy",

        "less":
            f"data/splits_C4_less_repeat{r:02d}/less_score_dict.npy",

        "orient":
            f"data/splits_C4_orient_repeat{r:02d}_epoch3/orient_score_dict.npy",

        "gradmatch":
            f"data/splits_C4_gradmatch_repeat{r:02d}/gradmatch_score_dict.npy",

        "craig":
            f"data/splits_C4_craig_repeat{r:02d}/craig_full_rank_score_dict.npy",

        "kmeans":
            f"data/splits_C4_kmeans_repeat{r:02d}/kmeans_feature_score_dict.npy",

        "kcenter":
            f"data/splits_C4_kcenter_repeat{r:02d}/kcenter_full_rank_score_dict.npy",

        "diversity":
            f"data/splits_C4_diversity_repeat{r:02d}/diversity_full_rank_score_dict.npy",
    }

    # -----------------------------
    # Load all
    # -----------------------------
    score_dicts = {}
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
    # Normalize all
    # -----------------------------
    norm_dicts = {}

    print("\n📊 Rank-normalizing...")

    for name, d in score_dicts.items():
        norm_dicts[name] = rank_normalize(d)
        print(f"Normalized {name}")

    # -----------------------------
    # Save
    # -----------------------------
    out_dir = f"data/splits_C4_mix_scores_multi/repeat{r:02d}"
    os.makedirs(out_dir, exist_ok=True)

    for name, d in norm_dicts.items():
        np.save(
            os.path.join(out_dir, f"{name}_norm_dict.npy"),
            d,
            allow_pickle=True
        )

    print(f"\n🎉 Saved 7D normalized dicts → {out_dir}")


if __name__ == "__main__":
    main()
