#!/usr/bin/env python3
import os
from collections import Counter

# --------------------------------------------
# 1. Load site → subject list mapping from TXT
# --------------------------------------------
def load_site_lists(collection_dir):
    site_to_subjects = {}

    for fname in os.listdir(collection_dir):
        if fname.endswith(".txt"):
            site = fname.replace(".txt", "")
            fpath = os.path.join(collection_dir, fname)

            with open(fpath, "r") as f:
                subjects = [line.strip() for line in f if line.strip()]
            
            site_to_subjects[site] = set(subjects)

    print(f"Loaded {len(site_to_subjects)} site lists.")
    return site_to_subjects


# --------------------------------------------
# 2. Detect site of a subject
# --------------------------------------------
def detect_site(subject_id, site_lists):
    for site, sset in site_lists.items():
        if subject_id in sset:
            return site
    return "UNKNOWN"


# --------------------------------------------
# 3. Analyze one subset file
# --------------------------------------------
def analyze_split(split_file, site_lists):
    with open(split_file, "r") as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("\n==============================")
    print(f"Analyzing: {split_file}")
    print("==============================")

    site_counts = Counter()

    for sid in subjects:
        site = detect_site(sid, site_lists)
        site_counts[site] += 1

    total = sum(site_counts.values())

    print("\nSite Distribution:")
    for site, cnt in site_counts.items():
        print(f"{site:25s} {cnt:4d} ({cnt/total*100:.1f}%)")

    print(f"\nTotal subjects: {total}")


# --------------------------------------------
# 4. MAIN
# --------------------------------------------
if __name__ == "__main__":

    # Your collections folder with site txt files
    COLLECTION_DIR = "data/002_BraTS21/collections"

    site_lists = load_site_lists(COLLECTION_DIR)

    SPLIT_FILES = [
        "data/splits_21_random/random_15T/train_subjects.txt",
        "data/splits_21_rdsplus/rdsplus_15T/train_subjects.txt",
        "data/splits_21_less_softmax_mean/less_15T/train_subjects.txt",
        "data/splits_21_less_softmax_kmeans/less_15T/train_subjects.txt",
    ]

    for sf in SPLIT_FILES:
        if os.path.isfile(sf):
            analyze_split(sf, site_lists)
        else:
            print(f"⚠️ Not found: {sf}")
