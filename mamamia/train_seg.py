#!/usr/bin/env python3
"""
Fine-tune EfficientViT on MAMA-MIA DCE-MRI data using nnUNet preprocessed files.

Loads .npy files (3 channels: pre, post1, post2) and _seg.npy masks.
Extracts 2D slices for training.

Usage:
    python -m mamamia.train_seg \
        --preprocessed-dir /path/to/nnUNet_preprocessed \
        --dataset DatasetXXX_NAME \
        --train-cases CASE_01 CASE_02 ... \
        --val-cases CASE_A CASE_B ... \
        --run-name my_experiment
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EFFICIENTVIT_DIR = PROJECT_ROOT / "models" / "efficientvit"
if EFFICIENTVIT_DIR.exists() and str(EFFICIENTVIT_DIR) not in sys.path:
    sys.path.append(str(EFFICIENTVIT_DIR))

from efficientvit.seg_model_zoo import create_efficientvit_seg_model


# Model settings
CROP_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", type=Path, required=True,
                        help="Path to nnUNet_preprocessed directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="nnUNet dataset name")
    parser.add_argument("--extra-datasets", type=str, nargs="*", default=[],
                        help="Additional datasets to search for cases")
    parser.add_argument("--train-cases", type=str, nargs="+", required=True,
                        help="Train case IDs")
    parser.add_argument("--val-cases", type=str, nargs="+", required=True,
                        help="Validation case IDs")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "models/efficientvit/assets/checkpoints/efficientvit_seg/efficientvit_seg_l1_ade20k.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "training_runs/efficientvit_mamamia")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tumor-pixels", type=int, default=100,
                        help="Minimum tumor pixels in slice to include")
    parser.add_argument("--foreground-oversample", type=float, default=None,
                        help="Fraction of samples from tumor slices (e.g. 0.33). "
                             "If set, includes ALL slices and uses weighted sampling.")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE,
                        help="Resize slices to this size (default: 512)")
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from last.pt checkpoint")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MamamiaSliceDataset(Dataset):
    """Dataset that loads 2D slices from nnUNet preprocessed 3D volumes."""

    def __init__(
        self,
        data_dirs: List[Path],
        case_ids: List[str],
        augment: bool = False,
        min_tumor_pixels: int = 100,
        crop_size: int = CROP_SIZE,
        include_all: bool = False,
    ):
        self.data_dirs = data_dirs
        self.augment = augment
        self.crop_size = crop_size
        self.slices: List[Tuple[str, int]] = []  # (case_id, slice_idx)
        self.is_foreground: List[bool] = []  # whether slice has tumor
        self.case_to_dir = {}  # Map case_id to its data directory

        # Collect all valid slices
        for case_id in case_ids:
            # Find which data_dir contains this case
            data_file = None
            seg_file = None
            case_dir = None
            for data_dir in data_dirs:
                df = data_dir / f"{case_id}.npy"
                sf = data_dir / f"{case_id}_seg.npy"
                if df.exists() and sf.exists():
                    data_file = df
                    seg_file = sf
                    case_dir = data_dir
                    break

            if data_file is None or seg_file is None:
                print(f"Warning: Missing files for {case_id}")
                continue

            self.case_to_dir[case_id] = case_dir

            # Load segmentation to find valid slices
            seg = np.load(seg_file)  # Shape: [1, D, H, W] or [D, H, W]
            if seg.ndim == 4:
                seg = seg[0]  # Remove channel dim

            for slice_idx in range(seg.shape[0]):
                tumor_pixels = (seg[slice_idx] > 0).sum()
                has_tumor = tumor_pixels >= min_tumor_pixels
                if include_all or has_tumor:
                    self.slices.append((case_id, slice_idx))
                    self.is_foreground.append(has_tumor)

        n_fg = sum(self.is_foreground)
        n_bg = len(self.slices) - n_fg
        print(f"Loaded {len(self.slices)} slices from {len(case_ids)} cases "
              f"({n_fg} tumor, {n_bg} non-tumor)")

    def __len__(self):
        return len(self.slices)

    def _normalize_and_resize(self, img: np.ndarray, mask: np.ndarray):
        """Normalize image channels and resize to crop_size."""
        # img shape: [C, H, W], mask shape: [H, W]

        # Resize using torch
        img_t = torch.from_numpy(img).float().unsqueeze(0)  # [1, C, H, W]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        img_t = F.interpolate(img_t, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
        mask_t = F.interpolate(mask_t, size=(self.crop_size, self.crop_size), mode='nearest')

        img_t = img_t.squeeze(0)  # [C, H, W]
        mask_t = mask_t.squeeze(0).squeeze(0)  # [H, W]

        # Convert to binary: 0 = background, 1 = tumor (anything > 0)
        mask_t = (mask_t > 0).long()

        # Normalize each channel to [0, 1] then apply ImageNet normalization
        for c in range(img_t.shape[0]):
            ch = img_t[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch = (ch - ch_min) / (ch_max - ch_min)
            img_t[c] = (ch - MEAN[c]) / STD[c]

        return img_t, mask_t.long()

    def _augment(self, img: torch.Tensor, mask: torch.Tensor):
        """Apply random augmentations."""
        if not self.augment:
            return img, mask

        # Random horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])

        # Random vertical flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])

        return img, mask

    def __getitem__(self, idx):
        case_id, slice_idx = self.slices[idx]
        data_dir = self.case_to_dir[case_id]

        # Load data
        data = np.load(data_dir / f"{case_id}.npy")  # [C, D, H, W]
        seg = np.load(data_dir / f"{case_id}_seg.npy")  # [1, D, H, W] or [D, H, W]

        if seg.ndim == 4:
            seg = seg[0]

        # Extract slice
        img_slice = data[:, slice_idx, :, :]  # [C, H, W]
        mask_slice = seg[slice_idx, :, :]  # [H, W]

        # Normalize and resize
        img_t, mask_t = self._normalize_and_resize(img_slice, mask_slice)

        # Augment
        img_t, mask_t = self._augment(img_t, mask_t)

        return {
            "image": img_t,
            "mask": mask_t,
            "case_id": case_id,
            "slice_idx": slice_idx,
        }


def load_model(checkpoint: Path, device: torch.device, num_classes: int = 2) -> nn.Module:
    """Load EfficientViT and modify for binary segmentation."""
    model = create_efficientvit_seg_model(
        name="efficientvit-seg-l1-ade20k",
        pretrained=False,
    )

    # Load pretrained weights
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            payload = state_dict["state_dict"]
        elif "model_state" in state_dict:
            payload = state_dict["model_state"]
        else:
            payload = state_dict
    else:
        payload = state_dict

    # Load weights (may have different head)
    model.load_state_dict(payload, strict=False)

    # Replace final classification layer for binary segmentation (2 classes)
    # The head is a DAGBlock with outputs["segout"] containing the final conv
    if hasattr(model, 'head') and hasattr(model.head, 'outputs'):
        segout = model.head.outputs.get("segout")
        if segout is not None:
            # segout is OpSequential, last item is the classifier
            # Find and replace the last ConvLayer
            for i in range(len(segout.op_list) - 1, -1, -1):
                layer = segout.op_list[i]
                if hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
                    in_ch = layer.conv.in_channels
                    # Replace with new conv for num_classes
                    segout.op_list[i].conv = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
                    print(f"Replaced classifier: {in_ch} -> {num_classes} classes")
                    break

    model.to(device)
    return model


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss (matches nnUNet's default loss).

    Computes soft Dice only on the label classes actually present (0 and 1),
    regardless of how many output channels the model has.
    """

    def __init__(self, label_classes: int = 2, smooth: float = 1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.label_classes = label_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        # Soft Dice on only the label classes (first label_classes channels)
        probs = F.softmax(logits, dim=1)[:, :self.label_classes]
        targets_onehot = F.one_hot(targets, self.label_classes).permute(0, 3, 1, 2).float()

        dims = (0,) + tuple(range(2, len(probs.shape)))
        intersection = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return ce_loss + dice_loss


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dice score for binary segmentation."""
    pred = (pred > 0).float()
    target = (target > 0).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection / union).item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    dices = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(logits, masks)
            losses.append(loss.item())

            preds = logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                dices.append(dice_score(preds[i], masks[i]))

    return np.mean(losses), np.mean(dices)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data directories (primary + extra)
    data_dirs = []
    primary_dir = args.preprocessed_dir / args.dataset / "nnUNetPlans_2d"
    if primary_dir.exists():
        data_dirs.append(primary_dir)

    for extra_ds in args.extra_datasets:
        extra_dir = args.preprocessed_dir / extra_ds / "nnUNetPlans_2d"
        if extra_dir.exists():
            data_dirs.append(extra_dir)

    if not data_dirs:
        raise FileNotFoundError(f"No data directories found")

    print(f"Loading data from: {data_dirs}")
    print(f"Train cases: {args.train_cases}")
    print(f"Val cases: {args.val_cases}")

    # Create datasets
    use_fg_oversample = args.foreground_oversample is not None
    train_dataset = MamamiaSliceDataset(
        data_dirs, args.train_cases, augment=True,
        min_tumor_pixels=args.min_tumor_pixels,
        include_all=use_fg_oversample,
        crop_size=args.crop_size,
    )
    val_dataset = MamamiaSliceDataset(
        data_dirs, args.val_cases, augment=False,
        min_tumor_pixels=args.min_tumor_pixels,
        include_all=use_fg_oversample,
        crop_size=args.crop_size,
    )

    # Build weighted sampler if foreground oversampling is requested
    train_sampler = None
    train_shuffle = True
    if use_fg_oversample:
        fg_frac = args.foreground_oversample
        n_fg = sum(train_dataset.is_foreground)
        n_bg = len(train_dataset) - n_fg
        if n_fg > 0 and n_bg > 0:
            w_fg = fg_frac / n_fg
            w_bg = (1.0 - fg_frac) / n_bg
            weights = [w_fg if fg else w_bg for fg in train_dataset.is_foreground]
            train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
            train_shuffle = False
            print(f"Foreground oversample: {fg_frac:.0%} tumor, {1-fg_frac:.0%} non-tumor")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")

    # Load model
    model = load_model(args.checkpoint, device)
    print("Model loaded")

    # Loss and optimizer
    criterion = DiceCELoss(label_classes=2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["train_slices"] = len(train_dataset)
    config["val_slices"] = len(val_dataset)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_dice = 0.0
    history = []

    if args.resume:
        last_ckpt = run_dir / "checkpoints" / "last.pt"
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            # Restore scheduler to correct position
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            # Restore best_dice from best.pt
            best_ckpt = run_dir / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                best_info = torch.load(best_ckpt, map_location=device, weights_only=False)
                best_dice = best_info.get("val_dice", 0.0)
            # Load existing history
            hist_path = run_dir / "training_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    history = json.load(f)
            print(f"Resumed from epoch {ckpt['epoch']}, best_dice={best_dice:.4f}")
        else:
            print("No last.pt found, starting fresh")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_dice={val_dice:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_dice": val_dice,
        }, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

        # Also save last and best
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_dice": val_dice,
        }, run_dir / "checkpoints" / "last.pt")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
            }, run_dir / "checkpoints" / "best.pt")
            print(f"  -> New best! Dice={best_dice:.4f}")

    # Save history
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
