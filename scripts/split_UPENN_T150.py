# #!/usr/bin/env python3
# import os
# import json
# import random

# # ---------- paths ----------
# COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
# OUTPUT_DIR = "/path/to/project/data/split_UPENN_T150"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# UPENN_FILE = "UPENN-GBM.txt"
# UPENN_ADD_FILE = "UPENN-GBM_Additional.txt"


# def load_list(path):
#     with open(path, "r") as f:
#         return [line.strip() for line in f.readlines() if line.strip()]


# # Load subjects
# upenn_main = load_list(os.path.join(COLLECTION_DIR, UPENN_FILE))
# upenn_add = load_list(os.path.join(COLLECTION_DIR, UPENN_ADD_FILE))

# all_subjects = upenn_main + upenn_add

# print(f"UPENN main: {len(upenn_main)}")
# print(f"UPENN additional: {len(upenn_add)}")
# print(f"Total UPENN: {len(all_subjects)}")
# print("Unique subjects:", len(set(all_subjects)))
# assert len(all_subjects) == 511, f"Expected 511, got {len(all_subjects)}"

# # --------------------------
# # Train / Val / Test sizes
# # --------------------------
# train_size = 100
# val_size = 50
# T = train_size + val_size
# test_size = 361

# assert T + test_size == 511

# random.seed(2025)
# subjects = all_subjects.copy()
# random.shuffle(subjects)

# train_subjects = sorted(subjects[:train_size])
# val_subjects = sorted(subjects[train_size:train_size + val_size])
# test_subjects = sorted(subjects[T:])

# assert len(test_subjects) == test_size

# # Write
# def write_list(path, lst):
#     with open(path, "w") as f:
#         for x in lst:
#             f.write(x + "\n")

# write_list(os.path.join(OUTPUT_DIR, "train_subjects.txt"), train_subjects)
# write_list(os.path.join(OUTPUT_DIR, "val_subjects.txt"), val_subjects)
# write_list(os.path.join(OUTPUT_DIR, "test_subjects.txt"), test_subjects)

# summary = {
#     "target_total": 511,
#     "train": len(train_subjects),
#     "val": len(val_subjects),
#     "test": len(test_subjects),
#     "seed": 2025
# }

# with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
#     json.dump(summary, f, indent=4)

# print("\n============================")
# print("  ✅ Finished UPENN split ")
# print("============================")
# print(summary)


#!/usr/bin/env python3
import os

# ---------- paths ----------
COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
OUTPUT_DIR = "/path/to/project/data/splits_UPENN_source"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target collections
UPENN_FILES = [
    "UPENN-GBM.txt",
    "UPENN-GBM_Additional.txt",
]

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

    if fname in UPENN_FILES:
        target_subjects.extend(subjects)

all_subjects = set(all_subjects)
target_subjects = set(target_subjects)

print(f"Total BraTS subjects: {len(all_subjects)}")
print(f"UPENN target subjects: {len(target_subjects)}")

source_subjects = sorted(all_subjects - target_subjects)

print(f"Source subjects: {len(source_subjects)}")

assert len(source_subjects) == len(all_subjects) - len(target_subjects)

# Write
with open(os.path.join(OUTPUT_DIR, "train_subjects.txt"), "w") as f:
    for s in source_subjects:
        f.write(s + "\n")

print("\n==============================")
print("  ✅ splits_UPENN_source done")
print("==============================")
