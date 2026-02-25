import os
import numpy as np

base_dir = "./data/splits_TCGA_rds"

ids_path = os.path.join(base_dir, "src_subject_ids.txt")
scores_path = os.path.join(base_dir, "rds_scores.npy")

# load
with open(ids_path) as f:
    subject_ids = [line.strip() for line in f]

scores = np.load(scores_path)

assert len(subject_ids) == len(scores)

# build dict
rds_score_dict = {
    subject_ids[i]: float(scores[i])
    for i in range(len(subject_ids))
}

# save
out_path = os.path.join(base_dir, "rds_score_dict.npy")
np.save(out_path, rds_score_dict, allow_pickle=True)

print(f"✅ Saved RDS score dict to {out_path}")
