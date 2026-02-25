#!/usr/bin/env python3
import os
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Utils
# -----------------------------
def load_grads_and_names(grad_dir, prefix):
    grads = np.load(os.path.join(grad_dir, f"{prefix}_less_grads.npy"))
    with open(os.path.join(grad_dir, f"{prefix}_less_names.txt")) as f:
        names = [line.strip() for line in f]
    assert grads.shape[0] == len(names)
    return grads, names


def extract_subject_id(slice_name):
    # "BraTS2021_00051_slice12" -> "BraTS2021_00051"
    return slice_name.split("_slice")[0]


# -----------------------------
# LESS scoring
# -----------------------------
def compute_less_slice_scores(src_grads, tgt_grads, beta=20.0, batch=1000):
    print("🔍 Computing slice-level LESS scores...")
    src_norm = src_grads / (np.linalg.norm(src_grads, axis=1, keepdims=True) + 1e-8)
    tgt_norm = tgt_grads / (np.linalg.norm(tgt_grads, axis=1, keepdims=True) + 1e-8)

    N = src_norm.shape[0]
    scores = np.zeros(N, dtype=np.float32)

    for start in tqdm(range(0, N, batch)):
        end = min(start + batch, N)
        sim = cosine_similarity(src_norm[start:end], tgt_norm)
        W = np.exp(beta * sim)
        W /= (W.sum(axis=1, keepdims=True) + 1e-8)
        scores[start:end] = (W * sim).sum(axis=1)

    return scores


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True, help="Repeat ID, e.g. 1")
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()

    grad_dir = f"results/less_gradients_C4_repeat{args.repeat:02d}"
    out_dir = f"data/splits_C4_less_repeat{args.repeat:02d}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Using gradients from: {grad_dir}")

    # Load gradients
    src_grads, src_names = load_grads_and_names(grad_dir, "brats21_source")
    tgt_grads, _ = load_grads_and_names(grad_dir, "brats21_target")

    # Slice-level LESS score
    slice_scores = compute_less_slice_scores(
        src_grads, tgt_grads,
        beta=args.beta, batch=args.batch
    )

    # Aggregate to subject-level (mean pooling)
    print("\n📦 Aggregating slice → subject (mean)...")
    subj_to_scores = defaultdict(list)
    for name, s in zip(src_names, slice_scores):
        pid = extract_subject_id(name)
        subj_to_scores[pid].append(s)

    less_score_dict = {
        pid: float(np.mean(scores))
        for pid, scores in subj_to_scores.items()
    }

    # Save
    out_path = os.path.join(out_dir, "less_score_dict.npy")
    np.save(out_path, less_score_dict, allow_pickle=True)

    print(f"💾 Saved LESS score dict → {out_path}")
    print(f"   Num subjects: {len(less_score_dict)}")


if __name__ == "__main__":
    main()
