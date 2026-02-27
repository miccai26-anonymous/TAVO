#!/usr/bin/env python3
import os
import json
import random

# ---------- paths ----------
COLLECTION_DIR = "/path/to/project/data/002_BraTS21/collections"
OUTPUT_DIR = "/path/to/project/data/split_T60"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TCGA_GBM_FILE = "TCGA-GBM.txt"
TCGA_LGG_FILE = "TCGA-LGG.txt"

def load_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Load cleaned target lists
tcga_gbm = load_list(os.path.join(COLLECTION_DIR, TCGA_GBM_FILE))
tcga_lgg = load_list(os.path.join(COLLECTION_DIR, TCGA_LGG_FILE))

print(f"TCGA-GBM: {len(tcga_gbm)}")
print(f"TCGA-LGG: {len(tcga_lgg)}")

target_total = len(tcga_gbm) + len(tcga_lgg)
assert target_total == 167, f"Expected 167 target subjects, got {target_total}"

# --------------------------
# Train / Val / Test sizes
# --------------------------
T = 60            # train + val
train_size = 40
val_size = 20

assert train_size + val_size == T

# Stratified sampling seed
random.seed(2025)

# Compute LGG/GBM proportion under T=60
n_lgg_T = round(T * len(tcga_lgg) / target_total)
n_gbm_T = T - n_lgg_T

print(f"Selected LGG={n_lgg_T}, GBM={n_gbm_T} for T=60")

# Shuffle lists
tcga_lgg_shuffled = tcga_lgg.copy()
tcga_gbm_shuffled = tcga_gbm.copy()
random.shuffle(tcga_lgg_shuffled)
random.shuffle(tcga_gbm_shuffled)

# choose T=60 stratified subjects
selected_lgg_T = tcga_lgg_shuffled[:n_lgg_T]
selected_gbm_T = tcga_gbm_shuffled[:n_gbm_T]

# Now split train / val
n_lgg_train = round(train_size * n_lgg_T / T)
n_gbm_train = train_size - n_lgg_train

n_lgg_val = n_lgg_T - n_lgg_train
n_gbm_val = n_gbm_T - n_gbm_train

# Compose sets
lgg_train = selected_lgg_T[:n_lgg_train]
gbm_train = selected_gbm_T[:n_gbm_train]

lgg_val = selected_lgg_T[n_lgg_train:n_lgg_train + n_lgg_val]
gbm_val = selected_gbm_T[n_gbm_train:n_gbm_train + n_gbm_val]

train_subjects = sorted(lgg_train + gbm_train)
val_subjects = sorted(lgg_val + gbm_val)

# Test = remaining (total 167 - 60)
remaining_target = sorted(
    set(tcga_lgg + tcga_gbm) - set(selected_lgg_T + selected_gbm_T)
)

assert len(remaining_target) == 107, f"Expected test=107, got {len(remaining_target)}"

# Write out files
def write_list(path, lst):
    with open(path, "w") as f:
        for x in lst:
            f.write(x + "\n")

write_list(os.path.join(OUTPUT_DIR, "train_subjects.txt"), train_subjects)
write_list(os.path.join(OUTPUT_DIR, "val_subjects.txt"), val_subjects)
write_list(os.path.join(OUTPUT_DIR, "test_subjects.txt"), remaining_target)

# Summary
summary = {
    "target_total": 167,
    "T": T,
    "T_lgg": n_lgg_T,
    "T_gbm": n_gbm_T,
    "train_total": len(train_subjects),
    "train_lgg": len(lgg_train),
    "train_gbm": len(gbm_train),
    "val_total": len(val_subjects),
    "val_lgg": len(lgg_val),
    "val_gbm": len(gbm_val),
    "test_total": len(remaining_target),
    "seed": 2025
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\n============================")
print("  ✅ Finished split_T60_new ")
print("============================")
print(summary)
