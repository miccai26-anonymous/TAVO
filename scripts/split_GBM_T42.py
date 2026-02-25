# #!/usr/bin/env python3
# import os
# import json
# import random

# COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
# OUTPUT_DIR = "/path/to/project/data/split_TCGA_GBM_T40"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# TARGET_FILE = "TCGA-GBM.txt"


# def load_list(path):
#     with open(path, "r") as f:
#         return [line.strip() for line in f.readlines() if line.strip()]


# subjects = load_list(os.path.join(COLLECTION_DIR, TARGET_FILE))

# print(f"TCGA-GBM total: {len(subjects)}")
# assert len(subjects) == 102  # adjust if count differs

# train_size = 27
# val_size = 15
# T = train_size + val_size
# test_size = len(subjects) - T

# random.seed(2025)
# random.shuffle(subjects)

# train_subjects = sorted(subjects[:train_size])
# val_subjects = sorted(subjects[train_size:T])
# test_subjects = sorted(subjects[T:])

# def write_list(path, lst):
#     with open(path, "w") as f:
#         for x in lst:
#             f.write(x + "\n")

# write_list(os.path.join(OUTPUT_DIR, "train_subjects.txt"), train_subjects)
# write_list(os.path.join(OUTPUT_DIR, "val_subjects.txt"), val_subjects)
# write_list(os.path.join(OUTPUT_DIR, "test_subjects.txt"), test_subjects)

# summary = {
#     "target_total": len(subjects),
#     "train": train_size,
#     "val": val_size,
#     "test": len(test_subjects),
#     "seed": 2025
# }

# with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
#     json.dump(summary, f, indent=4)

# print(summary)


# ==============================


#!/usr/bin/env python3
import os

# ---------- paths ----------
COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
OUTPUT_DIR = "/path/to/project/data/splits_TCGA_GBM_source"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FILE = "TCGA-GBM.txt"


def load_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ----------------------------
# Load all collections
# ----------------------------
all_subjects = []
target_subjects = []

for fname in os.listdir(COLLECTION_DIR):
    if not fname.endswith(".txt"):
        continue

    full_path = os.path.join(COLLECTION_DIR, fname)
    subjects = load_list(full_path)

    all_subjects.extend(subjects)

    if fname == TARGET_FILE:
        target_subjects.extend(subjects)

all_subjects = set(all_subjects)
target_subjects = set(target_subjects)

print(f"Total BraTS subjects: {len(all_subjects)}")
print(f"TCGA-GBM target subjects: {len(target_subjects)}")

source_subjects = sorted(all_subjects - target_subjects)

print(f"Source subjects: {len(source_subjects)}")

assert len(source_subjects) == len(all_subjects) - len(target_subjects)

# Write
with open(os.path.join(OUTPUT_DIR, "train_subjects.txt"), "w") as f:
    for s in source_subjects:
        f.write(s + "\n")

print("\n===================================")
print("  ✅ splits_TCGA_GBM_source done")
print("===================================")
