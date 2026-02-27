# #!/usr/bin/env python3
# import os
# import json
# import random

# # ---------- paths ----------
# COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
# OUTPUT_DIR = "/path/to/project/data/split_C5_T22"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# TARGET_FILE = "Collection_5.txt"


# def load_list(path):
#     with open(path, "r") as f:
#         return [line.strip() for line in f.readlines() if line.strip()]


# c5 = load_list(os.path.join(COLLECTION_DIR, TARGET_FILE))

# print(f"C5 total: {len(c5)}")
# assert len(c5) == 47

# # --------------------------
# # Train / Val / Test sizes
# # --------------------------
# train_size = 12
# val_size = 10
# T = train_size + val_size
# test_size = 25

# assert T + test_size == 47

# random.seed(2025)
# subjects = c5.copy()
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
#     "target_total": 47,
#     "train": 12,
#     "val": 10,
#     "test": 25,
#     "seed": 2025
# }

# with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
#     json.dump(summary, f, indent=4)

# print(summary)
# print("\n==============================")
# print("  ✅ split_C5_T22 done")
# print("==============================")

# ============================================================================================================

#!/usr/bin/env python3
import os

COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
OUTPUT_DIR = "/path/to/project/data/splits_C5_source"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FILE = "Collection_5.txt"


def load_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


all_subjects = []
target_subjects = []

for fname in os.listdir(COLLECTION_DIR):
    if not fname.endswith(".txt"):
        continue

    subjects = load_list(os.path.join(COLLECTION_DIR, fname))
    all_subjects.extend(subjects)

    if fname == TARGET_FILE:
        target_subjects.extend(subjects)

all_subjects = set(all_subjects)
target_subjects = set(target_subjects)

source_subjects = sorted(all_subjects - target_subjects)

with open(os.path.join(OUTPUT_DIR, "train_subjects.txt"), "w") as f:
    for s in source_subjects:
        f.write(s + "\n")

print(f"Source subjects: {len(source_subjects)}")
