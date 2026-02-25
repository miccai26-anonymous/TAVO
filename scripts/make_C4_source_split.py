#!/usr/bin/env python3
import os
import argparse
import glob

def read_subjects(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def write_subjects(path, subjects):
    with open(path, "w") as f:
        for s in subjects:
            f.write(f"{s}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--collections_dir",
        type=str,
        required=True,
        help="Path to collections dir, e.g. data/002_BraTS21/collections"
    )
    ap.add_argument(
        "--exclude_site",
        type=str,
        default="Collection_4",
        help="Site name to exclude from source"
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output dir, e.g. data/splits_C4_source"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_subjects = []
    excluded_subjects = []

    site_files = sorted(glob.glob(os.path.join(args.collections_dir, "*.txt")))
    assert len(site_files) > 0, "No collection txt files found."

    for fp in site_files:
        site = os.path.splitext(os.path.basename(fp))[0]
        subjects = read_subjects(fp)

        if site == args.exclude_site:
            excluded_subjects.extend(subjects)
        else:
            all_subjects.extend(subjects)

    all_subjects = sorted(list(set(all_subjects)))
    excluded_subjects = sorted(list(set(excluded_subjects)))

    print(f"Excluded site: {args.exclude_site}")
    print(f"Excluded subjects: {len(excluded_subjects)}")
    print(f"Source subjects (remaining): {len(all_subjects)}")

    out_path = os.path.join(args.out_dir, "train_subjects.txt")
    write_subjects(out_path, all_subjects)

    print(f"✅ Source split written to: {out_path}")

if __name__ == "__main__":
    main()
