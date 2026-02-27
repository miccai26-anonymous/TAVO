#!/usr/bin/env python3
import os
import argparse
import random


def load_subjects(src_file: str):
    with open(src_file, "r") as f:
        subs = [line.strip() for line in f if line.strip()]
    return subs


def dump_list(lst, out_txt: str):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for s in lst:
            f.write(s + "\n")
    print(f"✅ wrote {len(lst)} IDs -> {out_txt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make multi-seed random source splits (1T/5T/10T/15T).")

    ap.add_argument("--source_list", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])

    args = ap.parse_args()

    subjects = load_subjects(args.source_list)
    print(f"\nLoaded {len(subjects)} SOURCE subjects\n")

    multipliers = [1, 5, 10, 15]

    for seed in args.seeds:

        print(f"\n==========================")
        print(f" Seed = {seed}")
        print(f"==========================")

        random.seed(seed)
        shuffled = subjects.copy()
        random.shuffle(shuffled)

        for k in multipliers:

            n = min(k * args.T, len(shuffled))
            selected = shuffled[:n]

            split_name = f"seed{seed:02d}/random_{k}T"
            out_dir = os.path.join(args.out_root, split_name)
            out_txt = os.path.join(out_dir, "train_subjects.txt")

            dump_list(selected, out_txt)

    print("\n🎉 All multi-seed random splits generated.")
