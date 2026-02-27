# import os
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm

# def analyze_dataset(name, label_dir):
#     print(f"\n==============================")
#     print(f"📊 Dataset: {name}")
#     print(f"Label dir: {label_dir}")
#     print(f"==============================")

#     label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

#     stats = defaultdict(int)

#     liver_pixels = 0
#     tumor_pixels = 0

#     for lf in tqdm(label_files, desc=f"Processing {name}", ncols=100):
#         lbl = np.load(os.path.join(label_dir, lf))

#         has_liver = (lbl == 1).any()
#         has_tumor = (lbl == 2).any()

#         # ---- slice-level ----
#         stats["total_slices"] += 1

#         if not has_liver and not has_tumor:
#             stats["background_only"] += 1
#         elif has_liver and not has_tumor:
#             stats["liver_only"] += 1
#         elif has_tumor:
#             stats["tumor_slices"] += 1

#         # ---- pixel-level ----
#         liver_pixels += (lbl == 1).sum()
#         tumor_pixels += (lbl == 2).sum()

#     # ---- print summary ----
#     print(f"\nTotal slices      : {stats['total_slices']}")
#     print(f"Background only   : {stats['background_only']}")
#     print(f"Liver only        : {stats['liver_only']}")
#     print(f"Tumor slices      : {stats['tumor_slices']}")

#     print(f"\nPixel statistics:")
#     print(f"Liver pixels      : {liver_pixels:,}")
#     print(f"Tumor pixels      : {tumor_pixels:,}")

#     if liver_pixels + tumor_pixels > 0:
#         ratio = tumor_pixels / (liver_pixels + tumor_pixels)
#         print(f"Tumor pixel ratio : {ratio:.6f}")
#     else:
#         print("Tumor pixel ratio : N/A")

#     return stats

# if __name__ == "__main__":

#     datasets = {
#         "LiTS2017": "/path/to/project/data/LiTS/labelsTr",
#         "ATLAS": "/path/to/project/data/ATLAS/labelsTr",
#         "3Dircadb": "/path/to/project/data/3Dircadb/labelsTr",
#     }

#     for name, label_dir in datasets.items():
#         analyze_dataset(name, label_dir)

import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def check_dataset(name, label_dir):
    print(f"\n==============================")
    print(f"🔍 Dataset: {name}")
    print(f"Label dir: {label_dir}")
    print(f"==============================")

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

    # subject_id -> has_tumor
    subject_has_tumor = defaultdict(bool)
    subject_slice_count = defaultdict(int)

    for lf in tqdm(label_files, desc=f"Scanning {name}", ncols=100):
        slice_name = lf.replace(".npy", "")
        subject_id = slice_name.split("_slice")[0]

        lbl = np.load(os.path.join(label_dir, lf))

        subject_slice_count[subject_id] += 1
        if (lbl == 2).any():
            subject_has_tumor[subject_id] = True

    tumor_free_subjects = [
        sid for sid, has_t in subject_has_tumor.items() if not has_t
    ]

    print(f"\n📦 Total subjects        : {len(subject_slice_count)}")
    print(f"🧠 Subjects with tumor   : {len(subject_slice_count) - len(tumor_free_subjects)}")
    print(f"⚠️ Tumor-free subjects   : {len(tumor_free_subjects)}")

    if tumor_free_subjects:
        print("\n❗ Tumor-free subject IDs:")
        for sid in tumor_free_subjects:
            print(f"  - {sid} (slices={subject_slice_count[sid]})")
    else:
        print("\n✅ No tumor-free subjects found.")

    return tumor_free_subjects


if __name__ == "__main__":

    datasets = {
        "LiTS2017": "/path/to/project/data/LiTS/labelsTr",
        "ATLAS": "/path/to/project/data/ATLAS/labelsTr",
        "3Dircadb": "/path/to/project/data/3Dircadb/labelsTr",
    }

    for name, label_dir in datasets.items():
        check_dataset(name, label_dir)
