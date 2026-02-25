# #!/usr/bin/env python3
# import os
# import numpy as np
# from tqdm import tqdm

# label_dirs = [
#     "/path/to/project/data/002_BraTS21/labelsTr",
#     "/path/to/project/data/001_BraTS19/labelsTr",
# ]

# for label_dir in label_dirs:
#     fixed = 0
#     print(f"\n🧩 Cleaning labels in: {label_dir}")
#     for fname in tqdm(sorted(os.listdir(label_dir))):
#         if not fname.endswith(".npy"):
#             continue
#         path = os.path.join(label_dir, fname)
#         seg = np.load(path)
#         if np.any(seg < 0) or np.any(seg == 4):
#             seg[seg < 0] = 0
#             seg[seg == 4] = 3
#             np.save(path, seg)
#             fixed += 1

#     print(f"✅ Finished cleaning {fixed} files in {label_dir}")

#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm

label_dirs = {
    # "LiTS": "/path/to/project/data/LiTS/labelsTr",
    # "ATLAS": "/path/to/project/data/ATLAS/labelsTr",
    "3Dircadb": "/path/to/project/data/3Dircadb/labelsTr",
}

for name, label_dir in label_dirs.items():
    print(f"\n🧹 Cleaning dataset: {name}")
    fixed_files = 0
    fixed_pixels = 0

    for fname in tqdm(sorted(os.listdir(label_dir))):
        if not fname.endswith(".npy"):
            continue

        path = os.path.join(label_dir, fname)
        seg = np.load(path)

        if np.any(seg == -1):
            count = np.sum(seg == -1)
            seg[seg == -1] = 0
            np.save(path, seg)

            fixed_files += 1
            fixed_pixels += count

    print(f"✅ {name}: fixed {fixed_files} files, {fixed_pixels} pixels (-1 → 0)")
