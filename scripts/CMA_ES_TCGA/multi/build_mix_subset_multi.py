#!/usr/bin/env python3
import os
import argparse
import numpy as np

def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    args = ap.parse_args()

    norm_dir = "data/splits_TCGA_mix_scores"

    files = sorted([f for f in os.listdir(norm_dir) if f.endswith("_norm_dict.npy")])
    assert len(files) == len(args.weights)

    weights = softmax(args.weights)

    norm_dicts = [
        np.load(os.path.join(norm_dir, f), allow_pickle=True).item()
        for f in files
    ]

    keys = norm_dicts[0].keys()

    mix = {}
    for sid in keys:
        score = 0
        for w, d in zip(weights, norm_dicts):
            score += w * d[sid]
        mix[sid] = score

    ranked = sorted(mix.items(), key=lambda x: x[1], reverse=True)
    selected = [sid for sid, _ in ranked[:args.budget]]

    out_dir = "data/mix_output_TCGA"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train_subjects.txt"), "w") as f:
        f.write("\n".join(selected))

    print("✅ TCGA subset saved.")

if __name__ == "__main__":
    main()
