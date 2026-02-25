#!/usr/bin/env python3
"""Preprocess raw DCE-MRI data for segmentation.

Resamples to isotropic spacing, z-score normalizes using stats across
all DCE phases, and extracts first post-contrast channel.

Usage:
    python preprocess_raw.py \
        --src-images-dir /path/to/raw/imagesTr \
        --src-labels-dir /path/to/raw/labelsTr \
        --dst-images-dir /path/to/output/imagesTr \
        --dst-labels-dir /path/to/output/labelsTr \
        --case-ids CASE_01 CASE_02 ... \
        --target-spacing 1.0 1.0 1.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def compute_zscore_stats(src_img_dir, case_id, num_phases=3):
    """Compute mean/std across all DCE phases for z-score normalization."""
    all_values = []
    for phase in range(num_phases):
        img_path = src_img_dir / f"{case_id}_{phase:04d}.nii.gz"
        if img_path.exists():
            img_sitk = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(img_sitk)
            nonzero = arr[arr > 0]
            all_values.extend(nonzero.flatten().tolist())

    if len(all_values) > 0:
        return float(np.mean(all_values)), float(np.std(all_values))
    return 0.0, 1.0


def resample_sitk(image, new_spacing, interpolator=sitk.sitkBSpline):
    """Resample a SimpleITK image to new spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def zscore_normalize_sitk(image, mean, std):
    """Apply z-score normalization to a SimpleITK image."""
    arr = sitk.GetArrayFromImage(image).astype(np.float64)
    if std > 0:
        arr = (arr - mean) / std
    else:
        arr = arr - mean
    result = sitk.GetImageFromArray(arr.astype(np.float32))
    result.CopyInformation(image)
    return result


def process_case(case_id, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir,
                 target_spacing, post_contrast_phase=1, num_phases=3):
    """Process a single case."""
    src_img = src_img_dir / f"{case_id}_{post_contrast_phase:04d}.nii.gz"
    if not src_img.exists():
        print(f"Missing image: {src_img}")
        return None

    mean, std = compute_zscore_stats(src_img_dir, case_id, num_phases)
    img_sitk = sitk.ReadImage(str(src_img), sitk.sitkFloat32)
    img_normalized = zscore_normalize_sitk(img_sitk, mean, std)
    img_resampled = resample_sitk(img_normalized, target_spacing, interpolator=sitk.sitkBSpline)

    dst_img = dst_img_dir / f"{case_id}_0000.nii.gz"
    sitk.WriteImage(img_resampled, str(dst_img))

    if src_label_dir:
        src_label = src_label_dir / f"{case_id}.nii.gz"
        if src_label.exists():
            label_sitk = sitk.ReadImage(str(src_label), sitk.sitkUInt8)
            label_resampled = resample_sitk(label_sitk, target_spacing,
                                            interpolator=sitk.sitkNearestNeighbor)
            dst_label = dst_label_dir / f"{case_id}.nii.gz"
            sitk.WriteImage(label_resampled, str(dst_label))

    return case_id


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-images-dir", type=Path, required=True,
                        help="Source directory with raw NIfTI images")
    parser.add_argument("--src-labels-dir", type=Path, default=None,
                        help="Source directory with NIfTI labels (optional)")
    parser.add_argument("--dst-images-dir", type=Path, required=True,
                        help="Output directory for preprocessed images")
    parser.add_argument("--dst-labels-dir", type=Path, default=None,
                        help="Output directory for preprocessed labels")
    parser.add_argument("--case-ids", type=str, nargs="+", default=None,
                        help="Case IDs to process")
    parser.add_argument("--case-list", type=Path, default=None,
                        help="Text file with case IDs (one per line)")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target isotropic spacing (default: 1.0 1.0 1.0)")
    parser.add_argument("--post-contrast-phase", type=int, default=1,
                        help="Phase index for first post-contrast (default: 1)")
    parser.add_argument("--num-phases", type=int, default=3,
                        help="Number of DCE phases (default: 3)")
    args = parser.parse_args()

    if args.case_list and args.case_list.exists():
        case_ids = [l.strip() for l in args.case_list.read_text().splitlines() if l.strip()]
    elif args.case_ids:
        case_ids = args.case_ids
    else:
        raise ValueError("Must provide --case-ids or --case-list")

    args.dst_images_dir.mkdir(parents=True, exist_ok=True)
    if args.dst_labels_dir:
        args.dst_labels_dir.mkdir(parents=True, exist_ok=True)

    spacing = tuple(args.target_spacing)
    print(f"Processing {len(case_ids)} cases...")
    print(f"  Target spacing: {spacing}")

    for case_id in case_ids:
        result = process_case(
            case_id, args.src_images_dir, args.src_labels_dir,
            args.dst_images_dir, args.dst_labels_dir, spacing,
            args.post_contrast_phase, args.num_phases,
        )
        if result:
            print(f"  Processed {result}")

    print("Done.")


if __name__ == "__main__":
    main()
