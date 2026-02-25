#!/usr/bin/env python3
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

def read_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def dicom_mask_to_array(dicom_dir):
    """
    Read a DICOM mask series and return numpy array (Z,Y,X) bool
    """
    img = read_dicom_series(dicom_dir)
    arr = sitk.GetArrayFromImage(img)
    return arr > 0


def convert_case(pid, src_root, out_img, out_lbl):
    pid_num = pid.split(".")[1].zfill(3)

    ct_dir = os.path.join(src_root, pid, "PATIENT_DICOM")
    mask_root = os.path.join(src_root, pid, "MASKS_DICOM")

    liver_dir = os.path.join(mask_root, "liver")
    tumor_dirs = [
        os.path.join(mask_root, d)
        for d in os.listdir(mask_root)
        if d.startswith("livertumor")
    ]

    # ---------- Load CT ----------
    ct_img = read_dicom_series(ct_dir)
    ct_arr = sitk.GetArrayFromImage(ct_img)

    # ---------- Load liver ----------
    if not os.path.isdir(liver_dir):
        raise RuntimeError(f"{pid} has no liver mask")

    liver_mask = dicom_mask_to_array(liver_dir)

    # ---------- Load tumor (union) ----------
    tumor_mask = np.zeros_like(liver_mask, dtype=bool)
    for td in tumor_dirs:
        tumor_mask |= dicom_mask_to_array(td)

    if not tumor_mask.any():
        return False  # tumor-free, skip

    # ---------- Construct label ----------
    label = np.zeros_like(liver_mask, dtype=np.uint8)
    label[liver_mask] = 1
    label[tumor_mask] = 2   # tumor overrides liver

    # ---------- Save CT ----------
    ct_out = os.path.join(out_img, f"IRCAD_{pid_num}_0000.nii.gz")
    sitk.WriteImage(ct_img, ct_out)

    # ---------- Save label ----------
    lbl_img = sitk.GetImageFromArray(label.astype(np.uint8))
    lbl_img.CopyInformation(ct_img)

    lbl_out = os.path.join(out_lbl, f"IRCAD_{pid_num}.nii.gz")
    sitk.WriteImage(lbl_img, lbl_out)

    return True


def main(src_root, out_img, out_lbl):
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    patients = sorted([
        d for d in os.listdir(src_root)
        if d.startswith("3Dircadb1.")
    ])

    kept, skipped = 0, 0

    for pid in tqdm(patients, desc="Converting 3Dircadb", ncols=100):
        try:
            ok = convert_case(pid, src_root, out_img, out_lbl)
            if ok:
                kept += 1
                print(f"✅ {pid} → kept")
            else:
                skipped += 1
                print(f"⚠️ {pid} → tumor-free, skipped")
        except Exception as e:
            skipped += 1
            print(f"❌ {pid} → ERROR: {e}")

    print("\n==============================")
    print("📊 Conversion summary")
    print("==============================")
    print(f"Kept (tumor-positive): {kept}")
    print(f"Skipped              : {skipped}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert 3Dircadb to nnUNet format (liver + tumor only)"
    )
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--out_img", type=str, required=True)
    parser.add_argument("--out_lbl", type=str, required=True)
    args = parser.parse_args()

    main(args.src_root, args.out_img, args.out_lbl)
