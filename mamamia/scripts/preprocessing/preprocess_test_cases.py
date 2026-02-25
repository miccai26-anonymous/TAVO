#!/usr/bin/env python3
"""Preprocess test cases (nnUNet-style) and save as .npy + properties .json.

Avoids re-preprocessing for every model evaluation.

Pipeline per case:
  1. Load raw NIfTI channels (SimpleITK)
  2. Crop to nonzero bounding box
  3. Z-score normalize per channel
  4. Resample to target spacing
  5. Save preprocessed .npy and properties .json

Usage:
    python preprocess_test_cases.py \
        --test-images-dir /path/to/imagesTs \
        --test-labels-dir /path/to/labels \
        --plans-file /path/to/nnUNetPlans.json \
        --configuration 2d \
        --output-dir /path/to/preprocessed_test \
        --test-cases-file test_cases.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes

sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-images-dir", type=Path, required=True,
                        help="Directory with raw NIfTI test images")
    parser.add_argument("--test-labels-dir", type=Path, required=True,
                        help="Directory with NIfTI test labels")
    parser.add_argument("--plans-file", type=Path, required=True,
                        help="Path to nnUNetPlans.json for target spacing")
    parser.add_argument("--configuration", type=str, default="2d")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-cases", type=str, nargs="+", default=None)
    parser.add_argument("--test-cases-file", type=Path, default=None)
    parser.add_argument("--num-channels", type=int, default=3,
                        help="Number of input channels (default: 3)")
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def crop_to_nonzero(data):
    """Crop volume to nonzero bounding box."""
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        nonzero_mask = nonzero_mask | (data[c] != 0)
    nonzero_mask = binary_fill_holes(nonzero_mask)

    coords = np.argwhere(nonzero_mask)
    if len(coords) == 0:
        bbox = [(0, s) for s in data.shape[1:]]
        return data, bbox, data.shape[1:]

    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0) + 1
    bbox = [(int(lo), int(hi)) for lo, hi in zip(bbox_min, bbox_max)]
    original_shape = data.shape[1:]

    slicer = tuple([slice(None)] + [slice(lo, hi) for lo, hi in bbox])
    return data[slicer], bbox, original_shape


def zscore_normalize(data):
    """Z-score normalize per channel."""
    data = data.astype(np.float32)
    for c in range(data.shape[0]):
        mean = data[c].mean()
        std = data[c].std()
        data[c] = (data[c] - mean) / max(std, 1e-8)
    return data


def resample_volume(data, original_spacing, target_spacing):
    """Resample volume to target spacing."""
    from skimage.transform import resize as skimage_resize

    old_shape = np.array(data.shape[1:])
    old_spacing = np.array(original_spacing)
    new_spacing = np.array(target_spacing)
    new_shape = np.round(old_spacing / new_spacing * old_shape).astype(int)

    if np.all(new_shape == old_shape):
        return data

    C = data.shape[0]
    new_shape_tuple = tuple(int(s) for s in new_shape)
    resampled = np.zeros((C,) + new_shape_tuple, dtype=np.float32)
    for c in range(C):
        resampled[c] = skimage_resize(
            data[c], new_shape_tuple, order=3, mode='edge',
            anti_aliasing=False, preserve_range=True,
        )
    return resampled


def get_target_spacing(plans_file, configuration="2d"):
    """Read target spacing from nnUNet plans."""
    with open(plans_file) as f:
        plans = json.load(f)
    return plans["configurations"][configuration]["spacing"]


def preprocess_case(images_dir, case_id, target_spacing_2d, num_channels=3):
    """Preprocess a single case."""
    import SimpleITK as sitk

    channels = []
    sitk_img = None
    for ch_idx in range(num_channels):
        fpath = images_dir / f"{case_id}_{ch_idx:04d}.nii.gz"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing: {fpath}")
        img = sitk.ReadImage(str(fpath))
        if sitk_img is None:
            sitk_img = img
        channels.append(sitk.GetArrayFromImage(img))

    data = np.stack(channels, axis=0).astype(np.float32)
    spacing_sitk = sitk_img.GetSpacing()
    original_spacing = list(reversed(spacing_sitk))

    properties = {
        "original_spacing": original_spacing,
        "original_shape": list(data.shape[1:]),
    }

    data, bbox, shape_before_crop = crop_to_nonzero(data)
    properties["bbox"] = bbox
    properties["shape_before_crop"] = list(shape_before_crop)
    properties["shape_after_crop"] = list(data.shape[1:])

    data = zscore_normalize(data)

    target_spacing = [original_spacing[0], target_spacing_2d[0], target_spacing_2d[1]]
    data = resample_volume(data, original_spacing, target_spacing)
    properties["target_spacing"] = target_spacing
    properties["shape_after_resample"] = list(data.shape[1:])

    return data, properties


def load_label(labels_dir, case_id):
    """Load label volume."""
    import SimpleITK as sitk
    lbl_path = labels_dir / f"{case_id}.nii.gz"
    if not lbl_path.exists():
        raise FileNotFoundError(f"Missing: {lbl_path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_path))).astype(np.int32)


def _process_one_case(worker_args):
    images_dir, labels_dir, case_id, target_spacing_2d, out_dir, num_channels, idx, total = worker_args
    npy_path = out_dir / f"{case_id}.npy"
    props_path = out_dir / f"{case_id}_properties.json"
    label_path = out_dir / f"{case_id}_label.npy"

    try:
        data, properties = preprocess_case(images_dir, case_id, target_spacing_2d, num_channels)
        label = load_label(labels_dir, case_id)

        np.save(npy_path, data)
        np.save(label_path, label)
        with open(props_path, "w") as f:
            json.dump(properties, f, indent=2)

        print(f"  [{idx+1}/{total}] {case_id}: "
              f"orig {properties['original_shape']} -> preproc {list(data.shape[1:])}")

    except Exception as e:
        print(f"  [{idx+1}/{total}] {case_id}: ERROR - {e}")


def main():
    args = parse_args()

    if args.test_cases_file and args.test_cases_file.exists():
        with open(args.test_cases_file) as f:
            test_cases = [line.strip() for line in f if line.strip()]
    elif args.test_cases:
        test_cases = args.test_cases
    else:
        raise ValueError("Must provide --test-cases or --test-cases-file")

    target_spacing_2d = get_target_spacing(args.plans_file, args.configuration)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for case_id in test_cases:
        npy_path = out_dir / f"{case_id}.npy"
        props_path = out_dir / f"{case_id}_properties.json"
        label_path = out_dir / f"{case_id}_label.npy"
        if npy_path.exists() and props_path.exists() and label_path.exists():
            continue
        todo.append(case_id)

    print(f"Preprocessing {len(todo)} test cases ({len(test_cases) - len(todo)} already done)")
    print(f"  Images: {args.test_images_dir}")
    print(f"  Labels: {args.test_labels_dir}")
    print(f"  Target spacing (y, x): {target_spacing_2d}")
    print(f"  Output: {out_dir}")

    if args.num_workers > 1 and len(todo) > 1:
        from multiprocessing import Pool
        worker_args = [
            (args.test_images_dir, args.test_labels_dir, case_id,
             target_spacing_2d, out_dir, args.num_channels, i, len(todo))
            for i, case_id in enumerate(todo)
        ]
        with Pool(args.num_workers) as pool:
            pool.map(_process_one_case, worker_args)
    else:
        for i, case_id in enumerate(todo):
            _process_one_case((
                args.test_images_dir, args.test_labels_dir, case_id,
                target_spacing_2d, out_dir, args.num_channels, i, len(todo)
            ))

    print(f"\nDone. Preprocessed data saved to: {out_dir}")


if __name__ == "__main__":
    main()
