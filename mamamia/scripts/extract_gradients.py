#!/usr/bin/env python3
"""Extract per-case gradients and run gradient-based selection methods.

Steps:
1. Load pretrained nnUNet checkpoint (PlainConvUNet)
2. For each case: forward tumor slices, compute loss gradients, average
3. Use last-layer gradients (smaller, still informative)
4. Project to low dimension via CountSketch (preserves inner products)
5. Run LESS, GradMatch, CRAIG selections using projected gradients
6. Save gradient files + selection results

Usage:
    python extract_gradients.py \
        --checkpoint /path/to/checkpoint_final.pth \
        --data-dirs /path/to/preprocessed1 /path/to/preprocessed2 \
        --pool-list pool_cases.txt \
        --query-list query_cases.txt \
        --budget 250 --proj-dim 4096 --seed 42 \
        --output-dir /path/to/gradients \
        --selection-output-dir /path/to/selections
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from meta.fitness.nnunet_proxy import create_nnunet_2d_model, DCandCELoss


# CountSketch projection

def create_countsketch(grad_dim: int, proj_dim: int, seed: int = 0):
    """Create CountSketch projection arrays."""
    rng = np.random.RandomState(seed)
    hash_indices = rng.randint(0, proj_dim, size=grad_dim).astype(np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=grad_dim)
    return hash_indices, signs


def countsketch_project(grad_vec, hash_indices, signs, proj_dim):
    """Project a gradient vector using CountSketch."""
    projected = np.zeros(proj_dim, dtype=np.float32)
    np.add.at(projected, hash_indices, grad_vec * signs)
    return projected


# Model loading

def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device,
                               in_channels: int = 3, num_classes: int = 2):
    """Load pretrained nnUNet PlainConvUNet from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = create_nnunet_2d_model(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(ckpt['network_weights'])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def get_last_layer_params(model: nn.Module) -> List[nn.Parameter]:
    """Get parameters from last decoder stages + segmentation output layers."""
    params = []
    param_names = []
    all_named = list(model.named_parameters())
    for name, p in all_named:
        if any(key in name for key in ['seg_layers', 'decoder.stages.0', 'decoder.stages.1']):
            params.append(p)
            param_names.append(name)
    if not params:
        n_last = max(4, len(all_named) // 5)
        for name, p in all_named[-n_last:]:
            params.append(p)
            param_names.append(name)
    total = sum(p.numel() for p in params)
    print(f"  Using {len(params)} param groups ({total:,} params)")
    return params


# Gradient extraction

def extract_case_gradient(model, criterion, grad_params, data_dir, case_id, device,
                          hash_indices, signs, proj_dim, crop_size=256,
                          min_tumor_pixels=50, max_slices=20):
    """Extract average gradient for a single case, projected via CountSketch."""
    data_file = data_dir / f"{case_id}.npy"
    seg_file = data_dir / f"{case_id}_seg.npy"
    if not data_file.exists() or not seg_file.exists():
        return None

    data = np.load(data_file)
    seg = np.load(seg_file)
    if seg.ndim == 4:
        seg = seg[0]

    tumor_slices = [z for z in range(seg.shape[0]) if (seg[z] > 0).sum() >= min_tumor_pixels]
    if not tumor_slices:
        return None

    if len(tumor_slices) > max_slices:
        rng = np.random.RandomState(hash(case_id) % (2**31))
        tumor_slices = sorted(rng.choice(tumor_slices, max_slices, replace=False))

    proj_accum = np.zeros(proj_dim, dtype=np.float32)
    n_slices = 0

    for z in tumor_slices:
        img = data[:, z, :, :]
        mask = seg[z, :, :]

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

        img_t = F.interpolate(img_t, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        mask_t = F.interpolate(mask_t, size=(crop_size, crop_size), mode='nearest')
        mask_t = (mask_t.squeeze(0).squeeze(0) > 0).long().unsqueeze(0)

        for c in range(img_t.shape[1]):
            ch = img_t[0, c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch = (ch - ch_min) / (ch_max - ch_min)
            img_t[0, c] = (ch - 0.485) / 0.229

        img_t = img_t.to(device)
        mask_t = mask_t.to(device)

        model.zero_grad()
        outputs = model(img_t)
        if outputs.shape[-2:] != mask_t.shape[-2:]:
            outputs = F.interpolate(outputs, size=mask_t.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, mask_t)
        loss.backward()

        grads = []
        for p in grad_params:
            grads.append(p.grad.detach().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()))
        grad_vec = torch.cat(grads).numpy()

        proj_accum += countsketch_project(grad_vec, hash_indices, signs, proj_dim)
        n_slices += 1

    return proj_accum / n_slices


def find_case_dir(case_id: str, data_dirs: List[Path]) -> Optional[Path]:
    """Find which data directory contains a case."""
    for d in data_dirs:
        if (d / f"{case_id}.npy").exists():
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True,
                        help="Preprocessed data directories (nnUNetPlans_2d)")
    parser.add_argument("--pool-list", type=Path, required=True,
                        help="Text file with pool case IDs")
    parser.add_argument("--query-list", type=Path, required=True,
                        help="Text file with query case IDs")
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-slices", type=int, default=20)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-output-dir", type=Path, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model_from_checkpoint(args.checkpoint, device, args.in_channels, args.num_classes)
    criterion = DCandCELoss()
    grad_params = get_last_layer_params(model)
    grad_dim = sum(p.numel() for p in grad_params)
    print(f"  Gradient dim: {grad_dim:,} -> projected to {args.proj_dim}")

    hash_indices, signs = create_countsketch(grad_dim, args.proj_dim, seed=args.seed + 1000)

    pool_ids = [l.strip() for l in args.pool_list.read_text().splitlines() if l.strip()]
    query_ids = [l.strip() for l in args.query_list.read_text().splitlines() if l.strip()]
    print(f"Pool: {len(pool_ids)}, Query: {len(query_ids)}")

    # Extract pool gradients
    print(f"\nExtracting pool gradients...")
    pool_gradients, pool_ids_valid = [], []
    t0 = time.time()

    for i, case_id in enumerate(pool_ids):
        case_dir = find_case_dir(case_id, args.data_dirs)
        if case_dir is None:
            continue
        grad = extract_case_gradient(
            model, criterion, grad_params, case_dir, case_id, device,
            hash_indices, signs, args.proj_dim, args.crop_size, max_slices=args.max_slices)
        if grad is not None:
            pool_gradients.append(grad)
            pool_ids_valid.append(case_id)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pool_ids)}] {len(pool_ids_valid)} valid, {(i+1)/elapsed:.1f} cases/s")

    pool_gradients = np.array(pool_gradients, dtype=np.float32)
    print(f"Pool gradients: {pool_gradients.shape}")

    # Extract query gradients
    print(f"\nExtracting query gradients...")
    query_gradients, query_ids_valid = [], []
    for case_id in query_ids:
        case_dir = find_case_dir(case_id, args.data_dirs)
        if case_dir is None:
            continue
        grad = extract_case_gradient(
            model, criterion, grad_params, case_dir, case_id, device,
            hash_indices, signs, args.proj_dim, args.crop_size, max_slices=args.max_slices)
        if grad is not None:
            query_gradients.append(grad)
            query_ids_valid.append(case_id)

    query_gradients = np.array(query_gradients, dtype=np.float32)
    total_extract_time = time.time() - t0
    print(f"Query gradients: {query_gradients.shape}")
    print(f"Extraction time: {total_extract_time:.0f}s")

    # Save gradients
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "pool_gradients.npy", pool_gradients)
    np.save(args.output_dir / "query_gradients.npy", query_gradients)
    with open(args.output_dir / "pool_ids.json", "w") as f:
        json.dump(pool_ids_valid, f)
    with open(args.output_dir / "query_ids.json", "w") as f:
        json.dump(query_ids_valid, f)
    with open(args.output_dir / "info.json", "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "grad_dim_raw": grad_dim, "proj_dim": args.proj_dim,
            "projection": "countsketch",
            "n_pool": len(pool_ids_valid), "n_query": len(query_ids_valid),
            "max_slices": args.max_slices,
            "extraction_time_seconds": total_extract_time,
        }, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Run gradient-based selections
    print(f"\nRunning gradient-based selections...")
    from selection import get_selection_method
    args.selection_output_dir.mkdir(parents=True, exist_ok=True)

    methods_to_run = {"less": {"rank": 50}, "gradmatch": {"omp": True}, "craig": {}}

    for method_name, method_kwargs in methods_to_run.items():
        print(f"\n--- {method_name} (budget={args.budget}) ---")
        t0_sel = time.time()
        method = get_selection_method(method_name)(seed=args.seed, **method_kwargs)
        result = method.select(
            pool_ids=pool_ids_valid, budget=args.budget,
            gradients=pool_gradients, query_gradients=query_gradients)
        print(f"  Selected {len(result.selected)} in {time.time()-t0_sel:.1f}s")

        out_file = args.selection_output_dir / f"{method_name}_{args.budget}.json"
        with open(out_file, "w") as f:
            json.dump({
                "selected": result.selected, "scores": result.scores,
                "method": result.method, "budget": result.budget,
                "metadata": result.metadata,
                "grad_dim_raw": grad_dim, "proj_dim": args.proj_dim,
                "projection": "countsketch", "real_gradients": True,
            }, f, indent=2)
        print(f"  Saved to {out_file}")

    print(f"\nAll done! Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
