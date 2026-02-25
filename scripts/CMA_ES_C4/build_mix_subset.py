#!/usr/bin/env python3
import os, argparse, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)  # 250 / 500
    ap.add_argument("--w", type=float, required=True)     # in [0,1]
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    r = args.repeat
    w = float(args.w)
    assert 0.0 <= w <= 1.0

    norm_dir = f"data/splits_C4_mix_scores/repeat{r:02d}"
    rds  = np.load(os.path.join(norm_dir, "rds_norm_dict.npy"), allow_pickle=True).item()
    less = np.load(os.path.join(norm_dir, "less_norm_dict.npy"), allow_pickle=True).item()

    # mixed score
    mix = {sid: w * rds[sid] + (1.0 - w) * less[sid] for sid in rds.keys()}
    ranked = sorted(mix.items(), key=lambda x: x[1], reverse=True)
    selected = [sid for sid, _ in ranked[:args.budget]]

    os.makedirs(args.out_dir, exist_ok=True)
    out_txt = os.path.join(args.out_dir, "train_subjects.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(selected))

    print(out_txt)

if __name__ == "__main__":
    main()
