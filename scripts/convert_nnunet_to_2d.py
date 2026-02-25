# #!/usr/bin/env python3
# import os
# import csv
# import numpy as np
# import blosc2
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

# # ==============================================================
# # CONFIG
# # ==============================================================
# datasets = {
#     "Dataset001_BraTS19": "001_BraTS19",
#     "Dataset002_BraTS21": "002_BraTS21",
# }

# root_in = "/path/to/workspace/ANON_USER/BrainTumorSeg/nnunet_data/nnUNet_preprocessed"
# root_out = "/path/to/project/data"

# # Optional settings
# SAVE_PREVIEW = True      # Save per-subject preview image
# CSV_FILENAME = "slice_index.csv"

# # ==============================================================
# # HELPERS
# # ==============================================================
# def load_b2nd(path):
#     return blosc2.open(path)[:]

# def normalize(img):
#     img = img.astype(np.float32)
#     return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

# def visualize_modalities(img, seg, out_path, z):
#     """Save first-slice visualization (4 modalities + seg)"""
#     fig, axs = plt.subplots(1, 5, figsize=(18, 4))
#     titles = ["FLAIR", "T1", "T1CE", "T2", "Seg"]
#     for i in range(4):
#         axs[i].imshow(img[i, z], cmap="gray")
#         axs[i].set_title(f"{titles[i]} (z={z})")
#         axs[i].axis("off")
#     axs[4].imshow(seg[0, z], cmap="nipy_spectral")
#     axs[4].set_title("Seg")
#     axs[4].axis("off")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()

# # ==============================================================
# # MAIN
# # ==============================================================
# for dset_in, dset_out in datasets.items():
#     print(f"\n🚀 Converting full dataset: {dset_in}")
#     in_dir = join(root_in, dset_in, "nnUNetPlans_2d")
#     out_dir = join(root_out, dset_out)
#     images_out = join(out_dir, "imagesTr")
#     labels_out = join(out_dir, "labelsTr")
#     maybe_mkdir_p(images_out)
#     maybe_mkdir_p(labels_out)

#     # Prepare CSV index
#     csv_path = join(out_dir, CSV_FILENAME)
#     csv_file = open(csv_path, "w", newline="")
#     writer = csv.writer(csv_file)
#     writer.writerow(["subject", "slice_idx", "image_path", "label_path", "has_lesion"])

#     all_files = sorted([f for f in os.listdir(in_dir) if f.endswith(".b2nd") and "_seg" not in f])
#     print(f"📂 Found {len(all_files)} subjects in {dset_in}")

#     for f in tqdm(all_files, desc=f"Processing {dset_out}"):
#         base = f.replace(".b2nd", "")
#         subject_id = base
#         img_path = join(in_dir, f)
#         seg_path = join(in_dir, f"{base}_seg.b2nd")

#         if not os.path.exists(seg_path):
#             print(f"⚠️ Missing seg for {subject_id}, skipped.")
#             continue

#         # Load data
#         img = load_b2nd(img_path)     # (4, D, H, W)
#         seg = load_b2nd(seg_path)     # (1, D, H, W)
#         D = img.shape[1]

#         # Normalize
#         img = normalize(img)

#         # Generate per-slice arrays (H, W, 4)
#         for z in range(D):
#             img_slice = np.transpose(img[:, z, :, :], (1, 2, 0)).astype(np.float32)
#             seg_slice = seg[0, z, :, :].astype(np.int16)

#             # ✅ Fix label range
#             seg_slice[seg_slice < 0] = 0
#             seg_slice[seg_slice == 4] = 3

#             # Check if this slice contains lesion
#             has_lesion = int((seg_slice > 0).any())

#             out_img_path = join(images_out, f"{subject_id}_slice{z:03d}.npy")
#             out_seg_path = join(labels_out, f"{subject_id}_slice{z:03d}_seg.npy")

#             np.save(out_img_path, img_slice)
#             np.save(out_seg_path, seg_slice)

#             # Write index
#             writer.writerow([subject_id, z, out_img_path, out_seg_path, has_lesion])

#         # Save preview (optional)
#         if SAVE_PREVIEW:
#             mid = D // 2
#             out_png = join(out_dir, f"{subject_id}_preview.png")
#             visualize_modalities(img, seg, out_png, mid)

#     csv_file.close()
#     print(f"✅ Finished {dset_in}, CSV saved to: {csv_path}")

# print("\n🎉 All datasets converted successfully!")

## LiTS & 3Dircadb & ATLAS

#!/usr/bin/env python3
import os
import csv
import numpy as np
import blosc2
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

# ==============================================================
# CONFIG
# ==============================================================
datasets = {
    "Dataset012_3Dircadb": "3Dircadb",
    # "Dataset013_ATLAS": "ATLAS",
    # "Dataset011_LiTS": "LiTS",
}

root_in = "/path/to/workspace/ANON_USER/BrainTumorSeg/nnunet_data/nnUNet_preprocessed"
root_out = "/path/to/project/data"

SAVE_PREVIEW = True
CSV_FILENAME = "slice_index.csv"

# ==============================================================
# HELPERS
# ==============================================================
def load_b2nd(path):
    return blosc2.open(path)[:]

def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def visualize(img, seg, out_path, z):
    """
    img: (D, H, W) normalized image
    seg: (D, H, W) int segmentation
    """

    # ========= Fixed semantic colormap =========
    # 0 background -> black
    # 1 liver      -> light gray
    # 2 tumor      -> red
    cmap = mcolors.ListedColormap([
        "black",       # 0 background
        "lightgray",   # 1 liver
        "red"          # 2 tumor
    ])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # ---------- Image ----------
    axs[0].imshow(img[z], cmap="gray")
    axs[0].set_title(f"Image (z={z})")
    axs[0].axis("off")

    # ---------- Segmentation ----------
    im = axs[1].imshow(seg[z], cmap=cmap, norm=norm)
    axs[1].set_title("Segmentation\n(bg=black, liver=gray, tumor=red)")
    axs[1].axis("off")

    # ---------- Legend ----------
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="black", label="Background (0)"),
        Patch(facecolor="lightgray", label="Liver (1)"),
        Patch(facecolor="red", label="Tumor (2)"),
    ]
    axs[1].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# ==============================================================
# MAIN
# ==============================================================
for dset_in, dset_out in datasets.items():
    print(f"\n🚀 Converting {dset_in}")

    in_dir = join(root_in, dset_in, "nnUNetPlans_2d")
    out_dir = join(root_out, dset_out)
    images_out = join(out_dir, "imagesTr")
    labels_out = join(out_dir, "labelsTr")
    maybe_mkdir_p(images_out)
    maybe_mkdir_p(labels_out)

    csv_path = join(out_dir, CSV_FILENAME)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["subject", "slice_idx", "image_path", "label_path", "has_lesion"])

    all_imgs = sorted([f for f in os.listdir(in_dir) if f.endswith(".b2nd") and "_seg" not in f])
    print(f"📂 Found {len(all_imgs)} subjects")

    for f in tqdm(all_imgs, desc=f"Processing {dset_out}"):
        subject_id = f.replace(".b2nd", "")
        img_path = join(in_dir, f)
        seg_path = join(in_dir, f"{subject_id}_seg.b2nd")

        if not os.path.exists(seg_path):
            print(f"⚠️ Missing seg for {subject_id}, skipped")
            continue

        img = load_b2nd(img_path)    # (1, D, H, W)
        seg = load_b2nd(seg_path)    # (1, D, H, W)

        img = normalize(img[0])      # (D, H, W)
        seg = seg[0].astype(np.int16)

        D = img.shape[0]

        for z in range(D):
            img_slice = img[z][:, :, None]   # (H, W, 1)
            seg_slice = seg[z]

            has_lesion = int((seg_slice == 2).any())
            
            out_img = join(images_out, f"{subject_id}_slice{z:03d}.npy")
            out_seg = join(labels_out, f"{subject_id}_slice{z:03d}.npy")

            np.save(out_img, img_slice.astype(np.float32))
            np.save(out_seg, seg_slice.astype(np.int64))

            writer.writerow([subject_id, z, out_img, out_seg, has_lesion])

        if SAVE_PREVIEW:
            mid = D // 2
            out_png = join(out_dir, f"{subject_id}_preview.png")
            visualize(img, seg, out_png, mid)

    csv_file.close()
    print(f"✅ Finished {dset_in}, CSV: {csv_path}")

print("\n🎉 All datasets converted successfully!")
