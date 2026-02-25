#!/usr/bin/env python3
"""Evaluate EfficientViT models using 3D Dice with nnUNet-style preprocessing.

Loads preprocessed test data (.npy + properties .json) for efficient evaluation.

Pipeline per case:
  1. Load preprocessed .npy and properties
  2. Per-slice: min-max + ImageNet normalization -> resize to crop_size
  3. Run EfficientViT inference (batched, fp16)
  4. Reverse transforms (resample + uncrop) back to original space
  5. Compute 3D Dice against original labels

Usage:
    python evaluate_3d_dice_nnunet_preproc.py \
        --run-names run1 run2 \
        --preprocessed-test-dir /path/to/preprocessed_test \
        --training-runs-dir /path/to/training_runs \
        --test-cases-file cases.txt \
        --crop-size 512
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EFFICIENTVIT_DIR = PROJECT_ROOT / "models" / "efficientvit"
if str(EFFICIENTVIT_DIR) not in sys.path:
    sys.path.append(str(EFFICIENTVIT_DIR))

from efficientvit.seg_model_zoo import create_efficientvit_seg_model

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-names", type=str, nargs="+", required=True)
    parser.add_argument("--preprocessed-test-dir", type=Path, required=True,
                        help="Directory with preprocessed .npy and _properties.json files")
    parser.add_argument("--test-cases", type=str, nargs="+", default=None)
    parser.add_argument("--test-cases-file", type=Path, default=None)
    parser.add_argument("--training-runs-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Enable test-time augmentation (4-way flip)")
    parser.add_argument("--best-of-last", type=int, default=None,
                        help="Use best checkpoint from last N epochs")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def reverse_transform(pred_volume, properties):
    """Reverse nnUNet preprocessing transforms."""
    shape_after_crop = properties["shape_after_crop"]
    pred_t = torch.from_numpy(pred_volume).float().unsqueeze(0).unsqueeze(0)
    pred_t = F.interpolate(pred_t, size=list(shape_after_crop), mode='nearest')
    pred_cropped = pred_t.squeeze().numpy()

    shape_before_crop = properties["shape_before_crop"]
    pred_original = np.zeros(shape_before_crop, dtype=pred_cropped.dtype)
    bbox = properties["bbox"]
    slicer = tuple([slice(lo, hi) for lo, hi in bbox])
    pred_original[slicer] = pred_cropped
    return pred_original


def _get_checkpoint_num_classes(model_state):
    """Detect number of output classes from checkpoint."""
    pat = re.compile(r"^head\.output_ops\.0\.op_list\.(\d+)\.conv\.weight$")
    best_idx, best_key = None, None
    for k in model_state.keys():
        m = pat.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        if best_idx is None or idx > best_idx:
            best_idx = idx
            best_key = k
    if best_key is None:
        return None
    w = model_state.get(best_key)
    if w is None or getattr(w, "ndim", None) != 4 or tuple(w.shape[2:]) != (1, 1):
        return None
    return int(w.shape[0])


def _maybe_replace_classifier(model, num_classes):
    """Adjust classifier head if needed."""
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "output_ops"):
        return False
    try:
        out_op = head.output_ops[0]
        op_list = getattr(out_op, "op_list", None)
        if op_list is None or len(op_list) == 0:
            return False
        last = op_list[-1]
        conv = getattr(last, "conv", None)
        if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1):
            return False
        if conv.out_channels == num_classes:
            return True
        last.conv = nn.Conv2d(conv.in_channels, num_classes, kernel_size=1, bias=True)
        return True
    except Exception:
        return False


def load_model(checkpoint_path, device):
    """Load EfficientViT model from checkpoint."""
    model = create_efficientvit_seg_model(name="efficientvit-seg-l1-ade20k", pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = state_dict["model_state"] if isinstance(state_dict, dict) and "model_state" in state_dict else state_dict
    if isinstance(model_state, dict):
        ckpt_nc = _get_checkpoint_num_classes(model_state)
        if ckpt_nc is not None:
            _maybe_replace_classifier(model, ckpt_nc)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def normalize_slice(img_slice, crop_size):
    """Normalize and resize a single slice."""
    img_t = torch.from_numpy(img_slice).float().unsqueeze(0)
    img_t = F.interpolate(img_t, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
    img_t = img_t.squeeze(0)
    for c in range(img_t.shape[0]):
        ch = img_t[c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch = (ch - ch_min) / (ch_max - ch_min)
        img_t[c] = (ch - MEAN[c]) / STD[c]
    return img_t


def dice_3d(pred, target):
    """Compute 3D Dice score."""
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def _batched_inference(model, all_slices, device, batch_size=32):
    """Run batched inference with optional fp16."""
    all_logits = []
    use_amp = device.type == "cuda"
    ctx = torch.amp.autocast("cuda") if use_amp else torch.no_grad()
    with torch.no_grad(), ctx:
        for start in range(0, all_slices.shape[0], batch_size):
            batch = all_slices[start:start+batch_size].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu().float())
    return torch.cat(all_logits, dim=0)


def evaluate_volume(model, preprocessed_data, properties, device, crop_size,
                    batch_size=32, tta=False):
    """Evaluate a single volume."""
    C, D, H, W = preprocessed_data.shape
    slices = [normalize_slice(preprocessed_data[:, s, :, :], crop_size) for s in range(D)]
    all_slices = torch.stack(slices, dim=0)

    logits = _batched_inference(model, all_slices, device, batch_size)

    if tta:
        logits += _batched_inference(model, all_slices.flip(3), device, batch_size).flip(3)
        logits += _batched_inference(model, all_slices.flip(2), device, batch_size).flip(2)
        logits += _batched_inference(model, all_slices.flip(2, 3), device, batch_size).flip(2, 3)
        logits /= 4.0

    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    pred_volume = logits.argmax(dim=1).numpy().astype(np.float32)
    return reverse_transform(pred_volume, properties)


def _atomic_json_dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def main():
    args = parse_args()

    if args.test_cases_file and args.test_cases_file.exists():
        with open(args.test_cases_file) as f:
            test_cases = [line.strip() for line in f if line.strip()]
    elif args.test_cases:
        test_cases = args.test_cases
    else:
        raise ValueError("Must provide --test-cases or --test-cases-file")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Crop: {args.crop_size}, TTA: {args.tta}")
    print(f"Test cases: {len(test_cases)}")

    results = {}

    for run_name in args.run_names:
        print(f"\n{'='*50}\nEvaluating: {run_name}\n{'='*50}")

        ckpt_dir = args.training_runs_dir / run_name / "checkpoints"

        if args.best_of_last:
            epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"))
            if not epoch_files:
                print(f"  No epoch checkpoints in {ckpt_dir}")
                continue
            candidates = epoch_files[-args.best_of_last:]
            best_ckpt, best_val = None, -1.0
            for cf in candidates:
                info = torch.load(cf, map_location="cpu", weights_only=False)
                vd = info.get("val_dice", -1.0)
                if vd > best_val:
                    best_val = vd
                    best_ckpt = cf
            checkpoint_path = best_ckpt
        else:
            checkpoint_path = ckpt_dir / "best.pt"

        if not checkpoint_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            continue

        model = load_model(checkpoint_path, device)
        run_results = {}
        dices = []

        for case_id in test_cases:
            try:
                npy_path = args.preprocessed_test_dir / f"{case_id}.npy"
                props_path = args.preprocessed_test_dir / f"{case_id}_properties.json"
                label_path = args.preprocessed_test_dir / f"{case_id}_label.npy"

                if not npy_path.exists():
                    continue

                preproc_data = np.load(npy_path)
                with open(props_path) as f:
                    properties = json.load(f)
                label = np.load(label_path)

                pred = evaluate_volume(model, preproc_data, properties, device,
                                       args.crop_size, tta=args.tta)
                d = dice_3d(pred, label)

                run_results[case_id] = {
                    "dice_3d": d,
                    "num_slices": int(preproc_data.shape[1]),
                    "gt_volume": float((label > 0).sum()),
                    "pred_volume": float((pred > 0).sum()),
                }
                dices.append(d)
                print(f"  {case_id}: {d:.4f}")

                if args.output:
                    results[run_name] = run_results
                    _atomic_json_dump(args.output, results)

            except Exception as e:
                print(f"  {case_id}: ERROR - {e}")

        mean_d = np.mean(dices) if dices else 0.0
        std_d = np.std(dices) if dices else 0.0
        run_results["mean_dice_3d"] = mean_d
        run_results["std_dice_3d"] = std_d
        results[run_name] = run_results
        print(f"\n  Mean 3D Dice: {mean_d:.4f} +/- {std_d:.4f}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for rn, rr in sorted(results.items(), key=lambda x: x[1].get("mean_dice_3d", 0), reverse=True):
        print(f"  {rn:<40} {rr.get('mean_dice_3d', 0):.4f}")

    if args.output:
        _atomic_json_dump(args.output, results)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
