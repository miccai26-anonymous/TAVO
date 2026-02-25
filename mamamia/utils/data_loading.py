"""Data loading utilities for MAMAMIA dataset."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


# Constants
CROP_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_case_ids_from_json(json_path: Path, key: str = "selected") -> List[str]:
    """Load case IDs from a selection JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get(key, [])


def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings from JSONL file.

    Args:
        path: Path to JSONL file with {"image": str, "embedding": list} per line

    Returns:
        Tuple of (filenames, embeddings array of shape (N, D))
    """
    filenames = []
    vectors = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            filenames.append(record["image"])
            vectors.append(record["embedding"])

    if not vectors:
        raise RuntimeError(f"No embeddings found in {path}")

    embeddings = np.array(vectors, dtype=np.float32)
    return filenames, embeddings


def load_embeddings_dict(path: Path) -> Dict[str, np.ndarray]:
    """
    Load embeddings as a dictionary.

    Args:
        path: Path to JSONL file

    Returns:
        Dict mapping filename to embedding vector
    """
    filenames, embeddings = load_embeddings(path)
    return {name: embeddings[i] for i, name in enumerate(filenames)}


class MamamiaSliceDataset(Dataset):
    """
    Dataset that loads 2D slices from nnUNet preprocessed 3D volumes.

    Args:
        data_dirs: List of directories containing .npy files
        case_ids: List of case IDs to include
        augment: Whether to apply data augmentation
        min_tumor_pixels: Minimum tumor pixels to include a slice (tumor-only mode)
        crop_size: Output image size
        include_all: If True, include all slices (for foreground oversampling)
    """

    def __init__(
        self,
        data_dirs: List[Path],
        case_ids: List[str],
        augment: bool = False,
        min_tumor_pixels: int = 100,
        crop_size: int = CROP_SIZE,
        include_all: bool = False,
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.augment = augment
        self.crop_size = crop_size
        self.slices: List[Tuple[str, int]] = []
        self.is_foreground: List[bool] = []
        self.case_to_dir: Dict[str, Path] = {}

        for case_id in case_ids:
            data_file, seg_file, case_dir = self._find_case_files(case_id)
            if data_file is None:
                print(f"Warning: Missing files for {case_id}")
                continue

            self.case_to_dir[case_id] = case_dir
            seg = np.load(seg_file)
            if seg.ndim == 4:
                seg = seg[0]

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

    def _find_case_files(self, case_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Find data and segmentation files for a case (.npy or .npz)."""
        for data_dir in self.data_dirs:
            seg_file = data_dir / f"{case_id}_seg.npy"
            if not seg_file.exists():
                continue
            # Prefer .npy (faster), fall back to .npz
            data_file = data_dir / f"{case_id}.npy"
            if data_file.exists():
                return data_file, seg_file, data_dir
            npz_file = data_dir / f"{case_id}.npz"
            if npz_file.exists():
                return npz_file, seg_file, data_dir
        return None, None, None

    def __len__(self) -> int:
        return len(self.slices)

    def _normalize_and_resize(self, img: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize image channels and resize to crop_size."""
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

        img_t = F.interpolate(img_t, size=(self.crop_size, self.crop_size),
                              mode='bilinear', align_corners=False)
        mask_t = F.interpolate(mask_t, size=(self.crop_size, self.crop_size),
                               mode='nearest')

        img_t = img_t.squeeze(0)
        mask_t = (mask_t.squeeze(0).squeeze(0) > 0).long()

        # Normalize each channel
        for c in range(img_t.shape[0]):
            ch = img_t[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch = (ch - ch_min) / (ch_max - ch_min)
            img_t[c] = (ch - MEAN[c]) / STD[c]

        return img_t, mask_t

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations."""
        if not self.augment:
            return img, mask

        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])

        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, slice_idx = self.slices[idx]
        data_dir = self.case_to_dir[case_id]

        data_file = data_dir / f"{case_id}.npy"
        if data_file.exists():
            data = np.load(data_file)
        else:
            data = np.load(data_dir / f"{case_id}.npz")["data"]
        seg = np.load(data_dir / f"{case_id}_seg.npy")
        if seg.ndim == 4:
            seg = seg[0]

        img_slice = data[:, slice_idx, :, :]
        mask_slice = seg[slice_idx, :, :]

        img_t, mask_t = self._normalize_and_resize(img_slice, mask_slice)
        img_t, mask_t = self._augment(img_t, mask_t)

        return {
            "image": img_t,
            "mask": mask_t,
            "case_id": case_id,
            "slice_idx": slice_idx,
        }


def create_weighted_sampler(
    dataset: MamamiaSliceDataset,
    foreground_fraction: float = 0.33
) -> Optional[WeightedRandomSampler]:
    """
    Create a weighted sampler for foreground oversampling.

    Args:
        dataset: MamamiaSliceDataset with is_foreground attribute
        foreground_fraction: Target fraction of foreground samples

    Returns:
        WeightedRandomSampler or None if not applicable
    """
    n_fg = sum(dataset.is_foreground)
    n_bg = len(dataset) - n_fg

    if n_fg == 0 or n_bg == 0:
        return None

    w_fg = foreground_fraction / n_fg
    w_bg = (1.0 - foreground_fraction) / n_bg
    weights = [w_fg if fg else w_bg for fg in dataset.is_foreground]

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
