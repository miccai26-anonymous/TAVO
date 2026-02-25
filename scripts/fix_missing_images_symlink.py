#!/usr/bin/env python3
import os
from pathlib import Path

# === 路径配置 ===
source_images = "/path/to/project/data/001_BraTS19/imagesTr"
split_root = "/path/to/project/data/split_T30"

for split in ["train", "val", "test"]:
    target_dir = Path(split_root) / "imagesTr" / split
    label_dir = Path(split_root) / "labelsTr" / split

    if not label_dir.exists():
        print(f"⚠️  Skipping {split}: labelsTr/{split} not found.")
        continue

    os.makedirs(target_dir, exist_ok=True)
    count = 0

    for lbl_file in label_dir.glob("*.npy"):
        # 🩹 去掉 label 文件名中的 "_seg"
        base_name = lbl_file.name.replace("_seg", "")
        img_src = Path(source_images) / base_name
        img_dst = target_dir / base_name

        if img_src.exists():
            if img_dst.exists():
                img_dst.unlink()
            os.symlink(img_src, img_dst)
            count += 1
        else:
            print(f"🚫 Missing image for {base_name}")

    print(f"✅ [{split}] Created {count} symlinks under {target_dir}")
