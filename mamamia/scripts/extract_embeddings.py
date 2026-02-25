#!/usr/bin/env python3
"""Extract bottleneck embeddings from a trained nnUNet 2D encoder.

Hooks into the encoder's deepest stage to get fixed-size embeddings
for each 2D slice, then averages across slices per case.

Usage:
    python extract_embeddings.py \
        --model-folder /path/to/trained_model \
        --case-list cases.txt \
        --images-root /path/to/images \
        --output embeddings.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-folder", type=Path, required=True,
                        help="Path to trained nnUNet model folder")
    parser.add_argument("--case-list", type=Path, required=True,
                        help="Text file with one case ID per line")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="Root directory containing case folders with images")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file for embeddings")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    parser.add_argument("--num-channels", type=int, default=3,
                        help="Number of input channels (default: 3)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tumor-only", action="store_true",
                        help="Only use tumor-containing slices")
    parser.add_argument("--preprocessed-dirs", type=Path, nargs="+", default=[],
                        help="Preprocessed dirs to search for _seg.npy (for --tumor-only)")
    parser.add_argument("--min-tumor-pixels", type=int, default=10)
    return parser.parse_args()


class EmbeddingExtractor:
    """Extract bottleneck embeddings from 2D nnUNet encoder."""

    def __init__(self, model_folder: Path, fold: int = 0,
                 checkpoint: str = "checkpoint_final.pth", device: str = "cuda"):
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        self.device = torch.device(device)

        # Load plans
        with open(model_folder / "plans.json") as f:
            plans = json.load(f)
        config = plans["configurations"]["2d"]
        arch_kwargs = config["architecture"]["arch_kwargs"]

        with open(model_folder / "dataset.json") as f:
            dataset_json = json.load(f)
        num_input_channels = len(dataset_json["channel_names"])
        num_classes = len(dataset_json["labels"])
        print(f"Model: {num_input_channels} channels, {num_classes} classes")

        self.network = PlainConvUNet(
            input_channels=num_input_channels,
            n_stages=arch_kwargs["n_stages"],
            features_per_stage=arch_kwargs["features_per_stage"],
            conv_op=torch.nn.Conv2d,
            kernel_sizes=arch_kwargs["kernel_sizes"],
            strides=arch_kwargs["strides"],
            n_conv_per_stage=arch_kwargs["n_conv_per_stage"],
            num_classes=num_classes,
            n_conv_per_stage_decoder=arch_kwargs["n_conv_per_stage_decoder"],
            conv_bias=arch_kwargs["conv_bias"],
            norm_op=torch.nn.InstanceNorm2d,
            norm_op_kwargs=arch_kwargs["norm_op_kwargs"],
            dropout_op=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs=arch_kwargs["nonlin_kwargs"],
            deep_supervision=False,
        )

        checkpoint_path = model_folder / f"fold_{fold}" / checkpoint
        print(f"Loading weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        state_dict = ckpt["network_weights"]
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[10:] if k.startswith("_orig_mod.") else k
            new_state_dict[key] = v

        self.network.load_state_dict(new_state_dict, strict=False)
        self.network = self.network.to(self.device)
        self.network.eval()

        self.bottleneck_features = None
        encoder = self.network.encoder
        last_stage_idx = len(encoder.stages) - 1

        def hook_fn(module, input, output):
            self.bottleneck_features = output

        encoder.stages[last_stage_idx].register_forward_hook(hook_fn)

    def extract_embedding(self, image_data: np.ndarray,
                          tumor_slices=None) -> np.ndarray:
        """Extract embedding from (C, D, H, W) volume."""
        embeddings = []
        slice_indices = tumor_slices if tumor_slices is not None else range(image_data.shape[1])

        with torch.no_grad():
            for d in slice_indices:
                slice_data = image_data[:, d, :, :]
                slice_tensor = torch.from_numpy(slice_data[None]).float().to(self.device)
                _ = self.network(slice_tensor)
                feat = self.bottleneck_features
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                embeddings.append(feat.cpu().numpy())

        if not embeddings:
            return None

        embeddings = np.concatenate(embeddings, axis=0)
        avg = embeddings.mean(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        return avg


def load_case_images(case_id, images_root, num_channels=3):
    """Load multi-channel images for a case."""
    import SimpleITK as sitk

    case_dir = images_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    channels = []
    for ch in range(num_channels):
        img_path = case_dir / f"{case_id}_{ch:04d}.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = sitk.ReadImage(str(img_path))
        channels.append(sitk.GetArrayFromImage(img))

    return np.stack(channels, axis=0).astype(np.float32)


def zscore_normalize(image_data):
    """Z-score normalize per channel."""
    normalized = np.zeros_like(image_data, dtype=np.float32)
    for c in range(image_data.shape[0]):
        channel = image_data[c]
        nonzero_mask = channel > 0
        if nonzero_mask.any():
            mean = channel[nonzero_mask].mean()
            std = channel[nonzero_mask].std()
            normalized[c] = (channel - mean) / max(std, 1e-8)
        else:
            normalized[c] = channel
    return normalized


def find_tumor_slices(case_id, preprocessed_dirs, min_tumor_pixels=10):
    """Find tumor-containing slices from preprocessed segmentation."""
    for pdir in preprocessed_dirs:
        seg_path = pdir / f"{case_id}_seg.npy"
        if seg_path.exists():
            seg = np.load(seg_path, mmap_mode="r")
            tumor_mask = seg[0] > 0 if seg.ndim == 4 else seg > 0
            tumor_per_slice = tumor_mask.sum(axis=(1, 2))
            return np.where(tumor_per_slice >= min_tumor_pixels)[0]
    return None


def main():
    args = parse_args()

    case_ids = [l.strip() for l in args.case_list.read_text().splitlines() if l.strip()]
    print(f"Processing {len(case_ids)} cases")

    extractor = EmbeddingExtractor(
        model_folder=args.model_folder, fold=args.fold,
        checkpoint=args.checkpoint, device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    with args.output.open("w") as f:
        for case_id in tqdm(case_ids, desc="Extracting"):
            try:
                image_data = load_case_images(case_id, args.images_root, args.num_channels)
                image_data = zscore_normalize(image_data)

                tumor_slices = None
                if args.tumor_only:
                    tumor_slices = find_tumor_slices(
                        case_id, args.preprocessed_dirs, args.min_tumor_pixels)
                    if tumor_slices is not None and len(tumor_slices) == 0:
                        skipped += 1
                        continue

                embedding = extractor.extract_embedding(image_data, tumor_slices=tumor_slices)
                if embedding is None:
                    skipped += 1
                    continue

                record = {"image": case_id, "embedding": embedding.tolist()}
                f.write(json.dumps(record) + "\n")
                f.flush()

            except Exception as e:
                print(f"Error processing {case_id}: {e}")

    print(f"Embeddings saved to {args.output}")
    if skipped:
        print(f"Skipped {skipped} cases")


if __name__ == "__main__":
    main()
