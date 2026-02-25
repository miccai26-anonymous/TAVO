#!/usr/bin/env python3
"""Set up nnUNet dataset from selected cases.

Creates symlinks to preprocessed data for selected cases + target train/val cases.

Usage:
    python setup_nnunet_dataset.py \
        --selection-json /path/to/selection.json \
        --source-preprocessed-dirs /path/to/ext/nnUNetPlans_2d /path/to/target/nnUNetPlans_2d \
        --reference-dataset /path/to/nnUNet_preprocessed/DatasetXXX \
        --target-dataset /path/to/nnUNet_preprocessed/DatasetYYY \
        --target-train CASE_01 CASE_02 ... \
        --target-val CASE_A CASE_B ...
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-json", type=Path, required=True,
                        help="JSON file with selected_cases or selected list")
    parser.add_argument("--source-preprocessed-dirs", type=Path, nargs="+", required=True,
                        help="Preprocessed data directories to search for .npy files")
    parser.add_argument("--reference-dataset", type=Path, required=True,
                        help="Reference nnUNet preprocessed dataset (for plans/config files)")
    parser.add_argument("--target-dataset", type=Path, required=True,
                        help="Target dataset directory to create")
    parser.add_argument("--target-train", type=str, nargs="*", default=[],
                        help="Target domain train case IDs to include")
    parser.add_argument("--target-val", type=str, nargs="*", default=[],
                        help="Target domain validation case IDs")
    parser.add_argument("--plan-config", type=str, default="nnUNetPlans_2d",
                        help="Plan configuration subdirectory name")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Override dataset_name in plans")
    args = parser.parse_args()

    # Load selection
    with open(args.selection_json) as f:
        selection = json.load(f)
    selected_cases = selection.get("selected_cases", selection.get("selected", []))
    print(f"Selected {len(selected_cases)} external cases")

    # Setup target directories
    target_plans_dir = args.target_dataset / args.plan_config
    if target_plans_dir.exists():
        for f in target_plans_dir.iterdir():
            f.unlink()
    target_plans_dir.mkdir(parents=True, exist_ok=True)

    # Copy plans and config files from reference
    for fname in ["nnUNetPlans.json", "dataset.json", "dataset_fingerprint.json"]:
        src = args.reference_dataset / fname
        dst = args.target_dataset / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fname}")

    # Update dataset_name in plans
    if args.dataset_name:
        plans_path = args.target_dataset / "nnUNetPlans.json"
        with open(plans_path) as f:
            plans = json.load(f)
        plans["dataset_name"] = args.dataset_name
        with open(plans_path, "w") as f:
            json.dump(plans, f, indent=2)

    def find_and_link(case_id, search_dirs):
        """Find case in source dirs and create symlinks."""
        for src_dir in search_dirs:
            src_dir = src_dir / args.plan_config if not str(src_dir).endswith(args.plan_config) else src_dir
            found = False
            for ext in [".npy", "_seg.npy", ".npz", ".pkl"]:
                src = src_dir / f"{case_id}{ext}"
                dst = target_plans_dir / f"{case_id}{ext}"
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)
                    if ext == ".npy":
                        found = True
            return found
        return False

    # Link selected external cases
    linked = 0
    for case_id in selected_cases:
        if find_and_link(case_id, args.source_preprocessed_dirs):
            linked += 1
    print(f"Linked {linked} selected cases")

    # Link target train cases
    linked_train = 0
    for case_id in args.target_train:
        if find_and_link(case_id, args.source_preprocessed_dirs):
            linked_train += 1
    print(f"Linked {linked_train} target train cases")

    # Link target val cases
    linked_val = 0
    for case_id in args.target_val:
        if find_and_link(case_id, args.source_preprocessed_dirs):
            linked_val += 1
    print(f"Linked {linked_val} target val cases")

    total = linked + linked_train + linked_val
    print(f"Total: {total} cases ({linked} selected + {linked_train} train + {linked_val} val)")

    # Create splits_final.json
    train_keys = selected_cases + args.target_train
    val_keys = args.target_val
    splits = [{"train": train_keys, "val": val_keys}]
    with open(args.target_dataset / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Created splits_final.json: {len(train_keys)} train, {len(val_keys)} val")

    # Update numTraining in dataset.json
    dataset_json_path = args.target_dataset / "dataset.json"
    if dataset_json_path.exists():
        with open(dataset_json_path) as f:
            dataset_json = json.load(f)
        dataset_json["numTraining"] = len(train_keys) + len(val_keys)
        with open(dataset_json_path, "w") as f:
            json.dump(dataset_json, f, indent=2)

    print(f"\nDataset setup complete: {args.target_dataset.name}")


if __name__ == "__main__":
    main()
