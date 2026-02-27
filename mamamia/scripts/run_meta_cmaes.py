#!/usr/bin/env python3
"""
CMA-ES meta-optimization over data-valuation method weights.

Combines scores from multiple selection methods using a learned weight vector,
optimized via CMA-ES with an nnUNet proxy fitness (PlainConvUNet, short training).

Usage:
    python run_meta_cmaes.py \
        --pool-embeddings data/embeddings/pool_embeddings.jsonl \
        --query-embeddings data/embeddings/query_embeddings.jsonl \
        --data-dirs /path/to/nnUNet_preprocessed/DatasetXXX/nnUNetPlans_2d \
        --val-cases CASE_A CASE_B CASE_C \
        --budget 250 --generations 20 --popsize 8 --n-steps 500 \
        --output-dir outputs/meta/cmaes
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from selection import SELECTION_METHODS, get_selection_method
from utils.data_loading import load_embeddings, MamamiaSliceDataset

try:
    import cma
except ImportError:
    raise ImportError("pycma required. Install: pip install cma")


# ---------------------------------------------------------------------------
# CMA-ES optimizer (ask/tell interface around pycma)
# ---------------------------------------------------------------------------
class CMAESOptimizer:
    """CMA-ES optimizer for selection method weights."""

    def __init__(self, n_methods: int, seed: int = 42):
        self.n_methods = n_methods
        self.seed = seed
        self.es: Optional[cma.CMAEvolutionStrategy] = None
        self._solutions = []
        self._idx = 0
        self._fitnesses = []

    def initialize(self, sigma0=0.3, popsize=8, maxiter=100):
        x0 = np.ones(self.n_methods) / self.n_methods
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "seed": self.seed,
            "bounds": [0, 1],
            "maxiter": maxiter,
            "popsize": popsize,
            "verbose": -9,
        })

    def ask(self) -> np.ndarray:
        if not self._solutions or self._idx >= len(self._solutions):
            self._solutions = self.es.ask()
            self._idx = 0
            self._fitnesses = []
        w = np.array(self._solutions[self._idx])
        self._idx += 1
        w = np.maximum(w, 0)
        s = w.sum()
        return w / s if s > 0 else np.ones(self.n_methods) / self.n_methods

    def tell(self, weights: np.ndarray, fitness: float):
        self._fitnesses.append(-fitness)  # CMA-ES minimizes
        if self._idx >= len(self._solutions):
            self.es.tell(self._solutions, self._fitnesses)


# ---------------------------------------------------------------------------
# Weighted score combiner
# ---------------------------------------------------------------------------
def combine_scores(
    scores: Dict[str, Dict[str, float]],
    weights: np.ndarray,
    methods: List[str],
) -> Dict[str, float]:
    """Normalize each method's scores to [0,1], then weighted sum."""
    all_samples = sorted({s for m in scores.values() for s in m})
    n_samples = len(all_samples)
    n_methods = len(methods)
    mat = np.zeros((n_samples, n_methods))
    s2i = {s: i for i, s in enumerate(all_samples)}

    for j, method in enumerate(methods):
        if method not in scores:
            continue
        for sample, score in scores[method].items():
            mat[s2i[sample], j] = score

    for j in range(n_methods):
        lo, hi = mat[:, j].min(), mat[:, j].max()
        if hi > lo:
            mat[:, j] = (mat[:, j] - lo) / (hi - lo)
        else:
            mat[:, j] = 0.5

    w = np.maximum(weights, 0)
    s = w.sum()
    w = w / s if s > 0 else np.ones(n_methods) / n_methods

    combined = mat @ w
    return {all_samples[i]: float(combined[i]) for i in range(n_samples)}


# ---------------------------------------------------------------------------
# nnUNet proxy fitness evaluator
# ---------------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        n_classes = pred_soft.shape[1]
        target_oh = torch.zeros_like(pred_soft)
        target_oh.scatter_(1, target.unsqueeze(1), 1)
        dice_sum = 0.0
        for c in range(1, n_classes):
            p = pred_soft[:, c].flatten(1)
            t = target_oh[:, c].flatten(1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice_sum += (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice_sum.mean() / max(1, n_classes - 1)


class DCandCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()

    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)


def create_plainconv_unet(in_channels=3, num_classes=2):
    """Create PlainConvUNet matching nnUNet 2d config."""
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    return PlainConvUNet(
        input_channels=in_channels, n_stages=7,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512],
        conv_op=nn.Conv2d, kernel_sizes=[[3, 3]] * 7,
        strides=[[1, 1]] + [[2, 2]] * 6,
        n_conv_per_stage=[2] * 7, n_conv_per_stage_decoder=[2] * 6,
        conv_bias=True, norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        num_classes=num_classes,
    )


def compute_val_dice(model, val_loader, device):
    model.eval()
    total_inter = total_union = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            out = model(images)
            if out.shape[-2:] != masks.shape[-2:]:
                out = F.interpolate(out, masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = out.argmax(dim=1)
            total_inter += ((pred == 1) & (masks == 1)).sum().item()
            total_union += (pred == 1).sum().item() + (masks == 1).sum().item()
    model.train()
    return 2.0 * total_inter / total_union if total_union > 0 else 0.0


class NNUNetProxyFitness:
    """Train PlainConvUNet for N steps, return validation Dice as fitness."""

    def __init__(self, data_dirs, val_cases, batch_size=12, lr=1e-3,
                 crop_size=256, best_k=4):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.batch_size = batch_size
        self.lr = lr
        self.crop_size = crop_size
        self.best_k = best_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.val_dataset = MamamiaSliceDataset(
            data_dirs=self.data_dirs, case_ids=val_cases,
            augment=False, min_tumor_pixels=50, crop_size=crop_size,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        print(f"  Val: {len(self.val_dataset)} slices from {len(val_cases)} cases")

    def evaluate(self, selected_cases, n_steps):
        train_ds = MamamiaSliceDataset(
            data_dirs=self.data_dirs, case_ids=selected_cases,
            augment=True, min_tumor_pixels=50, crop_size=self.crop_size,
        )
        if len(train_ds) == 0:
            return {"fitness": float("-inf"), "val_dices": [], "train_losses": []}

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )

        model = create_plainconv_unet(in_channels=3, num_classes=2).to(self.device)
        criterion = DCandCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=3e-5)

        model.train()
        train_iter = iter(train_loader)
        train_losses, val_dices = [], []
        eval_every = max(1, n_steps // 10)

        for step in range(n_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            optimizer.zero_grad()
            out = model(images)
            if out.shape[-2:] != masks.shape[-2:]:
                out = F.interpolate(out, masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(out, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer.step()
            train_losses.append(loss.item())

            if (step + 1) % eval_every == 0:
                val_dices.append(compute_val_dice(model, self.val_loader, self.device))

        if val_dices:
            best_k = sorted(val_dices, reverse=True)[: self.best_k]
            fitness = float(np.median(best_k))
        else:
            fitness = float("-inf")

        del model, optimizer
        torch.cuda.empty_cache()

        return {
            "fitness": fitness, "val_dices": val_dices,
            "train_losses": train_losses,
            "final_val_dice": val_dices[-1] if val_dices else None,
            "n_train_slices": len(train_ds),
        }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def compute_all_scores(methods, pool_ids, pool_emb, query_emb, budget, seed=42):
    all_scores = {}
    for name in methods:
        print(f"  Computing scores: {name} ...")
        cls = get_selection_method(name)
        result = cls(seed=seed).select(
            pool_ids=pool_ids, budget=budget,
            embeddings=pool_emb, query_embeddings=query_emb,
        )
        all_scores[name] = result.scores
    return all_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DEFAULT_METHODS = [
    "rds", "less", "gradmatch", "kcenter", "diversity", "kmeans", "craig_proxy",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pool-embeddings", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True,
                        help="nnUNet preprocessed dirs (nnUNetPlans_2d)")
    parser.add_argument("--val-cases", type=str, nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--n-steps", type=int, default=500,
                        help="Proxy training steps per candidate evaluation")
    parser.add_argument("--fitness", type=str, default="median_val",
                        choices=["median_val", "best_val", "final_val"])
    parser.add_argument("--best-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/meta/cmaes"))
    args = parser.parse_args()

    print("=" * 60)
    print("CMA-ES Meta-Optimizer (nnUNet PlainConvUNet Proxy)")
    print("=" * 60)
    print(f"  Steps/eval: {args.n_steps}  Fitness: {args.fitness} (best_k={args.best_k})")
    print(f"  Generations: {args.generations}  Population: {args.popsize}")
    print(f"  Budget: {args.budget}  Crop: {args.crop_size}  Batch: {args.batch_size}")

    methods = [m for m in args.methods if m in SELECTION_METHODS]
    if len(methods) < len(args.methods):
        missing = set(args.methods) - set(methods)
        print(f"  Warning: Missing methods: {missing}")

    # Load embeddings
    print(f"\nLoading embeddings ...")
    pool_ids, pool_emb = load_embeddings(args.pool_embeddings)
    _, query_emb = load_embeddings(args.query_embeddings)
    print(f"  Pool: {len(pool_ids)}, Query: {query_emb.shape[0]}")

    # Compute selection scores
    print(f"\nComputing scores for {len(methods)} methods ...")
    scores = compute_all_scores(methods, pool_ids, pool_emb, query_emb,
                                args.budget, args.seed)

    # Fitness evaluator
    print(f"\nSetting up nnUNet proxy fitness evaluator ...")
    fitness_eval = NNUNetProxyFitness(
        data_dirs=args.data_dirs, val_cases=args.val_cases,
        batch_size=args.batch_size, lr=args.lr,
        crop_size=args.crop_size, best_k=args.best_k,
    )

    # Initialize CMA-ES
    optimizer = CMAESOptimizer(n_methods=len(methods), seed=args.seed)
    optimizer.initialize(sigma0=0.3, popsize=args.popsize, maxiter=args.generations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.fitness}_{args.seed}"

    # --- Resume from checkpoint ---
    checkpoint_path = args.output_dir / f"checkpoint_{tag}.json"
    history_path = args.output_dir / f"history_{tag}.json"

    best_weights = None
    best_fitness = float("-inf")
    all_history = []
    generation_summaries = []
    t_start = time.time()
    start_eval = 0
    global_eval_idx = 0
    total_evals = args.generations * args.popsize

    if checkpoint_path.exists() and history_path.exists():
        print(f"\n>>> Resuming from checkpoint ...")
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        with open(history_path) as f:
            all_history = json.load(f)

        completed_gens = ckpt["completed_generations"]
        bw = ckpt["best_weights"]
        best_weights = np.array([bw[m] for m in methods])
        best_fitness = ckpt["best_fitness"]
        generation_summaries = ckpt["generation_summaries"]

        # Replay to restore CMA-ES internal state
        print(f"  Replaying {len(all_history)} evaluations ...")
        for record in all_history:
            _ = optimizer.ask()
            optimizer.tell(np.array(record["weights"]), record["fitness"])

        start_eval = completed_gens * args.popsize
        global_eval_idx = len(all_history)
        print(f"  Resumed: gen {completed_gens}, {global_eval_idx} evals, best={best_fitness:.4f}")

    # --- Main loop ---
    print(f"\nRunning CMA-ES ({args.generations} generations) ...")
    current_gen_evals = []

    def save_checkpoint(gen):
        bw_norm = best_weights / best_weights.sum() if best_weights is not None and best_weights.sum() > 0 else best_weights
        ckpt = {
            "config": {
                "optimizer": "cmaes", "budget": args.budget,
                "generations": args.generations, "popsize": args.popsize,
                "methods": methods, "n_steps": args.n_steps,
                "fitness_type": args.fitness, "best_k": args.best_k, "seed": args.seed,
            },
            "best_weights": {m: float(best_weights[i]) for i, m in enumerate(methods)} if best_weights is not None else None,
            "best_weights_normalized": {m: float(bw_norm[i]) for i, m in enumerate(methods)} if bw_norm is not None else None,
            "best_fitness": float(best_fitness),
            "completed_generations": gen,
            "generation_summaries": generation_summaries,
            "total_time_seconds": time.time() - t_start,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        with open(history_path, "w") as f:
            json.dump(all_history, f, indent=2)

    for eval_idx in range(start_eval, total_evals):
        weights = optimizer.ask()
        gen = eval_idx // args.popsize + 1
        eval_in_gen = eval_idx % args.popsize + 1
        global_eval_idx += 1

        combined = combine_scores(scores, weights, methods)
        selected = [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])[:args.budget]]

        t0 = time.time()
        result = fitness_eval.evaluate(selected, args.n_steps)
        fitness = result["fitness"]
        elapsed = time.time() - t0

        optimizer.tell(weights, fitness)

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = weights.copy()

        record = {
            "generation": gen, "eval_in_generation": eval_in_gen,
            "global_eval_idx": global_eval_idx,
            "weights": weights.tolist(),
            "weights_dict": {m: float(weights[i]) for i, m in enumerate(methods)},
            "fitness": float(fitness),
            "val_dices": result["val_dices"],
            "final_val_dice": result.get("final_val_dice"),
            "n_train_slices": result.get("n_train_slices"),
            "elapsed_seconds": elapsed,
        }
        all_history.append(record)
        current_gen_evals.append(record)

        total_elapsed = time.time() - t_start
        eta = (total_elapsed / global_eval_idx) * (total_evals - global_eval_idx)
        print(f"  [{global_eval_idx}/{total_evals}] Gen {gen}.{eval_in_gen}: "
              f"fitness={fitness:.4f}, best={best_fitness:.4f} "
              f"[{elapsed:.0f}s, ETA {eta/60:.0f}min]")

        if eval_in_gen == args.popsize:
            gen_fitnesses = [e["fitness"] for e in current_gen_evals]
            sorted_gen = sorted(current_gen_evals, key=lambda x: x["fitness"], reverse=True)
            generation_summaries.append({
                "generation": gen,
                "best_fitness_this_gen": float(max(gen_fitnesses)),
                "global_best_fitness": float(best_fitness),
                "mean_fitness": float(np.mean(gen_fitnesses)),
                "std_fitness": float(np.std(gen_fitnesses)),
                "best_weights_this_gen": sorted_gen[0]["weights_dict"],
            })
            print(f"  --- Gen {gen}: best={max(gen_fitnesses):.4f}, "
                  f"mean={np.mean(gen_fitnesses):.4f} +/- {np.std(gen_fitnesses):.4f}, "
                  f"global_best={best_fitness:.4f}")
            save_checkpoint(gen)
            current_gen_evals = []

    # --- Final output ---
    best_w_norm = best_weights / best_weights.sum() if best_weights.sum() > 0 else best_weights
    print(f"\n{'='*60}")
    print(f"Optimization complete!  Best fitness: {best_fitness:.4f}")
    print(f"Best weights (normalized):")
    for m, w in zip(methods, best_w_norm):
        print(f"  {m}: {w:.4f}")

    combined = combine_scores(scores, best_weights, methods)
    selected = [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])[:args.budget]]

    results = {
        "config": {
            "optimizer": "cmaes", "budget": args.budget,
            "generations": args.generations, "popsize": args.popsize,
            "methods": methods, "n_steps": args.n_steps,
            "fitness_type": args.fitness, "best_k": args.best_k, "seed": args.seed,
        },
        "best_weights": {m: float(best_weights[i]) for i, m in enumerate(methods)},
        "best_weights_normalized": {m: float(best_w_norm[i]) for i, m in enumerate(methods)},
        "best_fitness": float(best_fitness),
        "selected_cases": selected,
        "generation_summaries": generation_summaries,
        "total_time_seconds": time.time() - t_start,
    }

    with open(args.output_dir / f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_dir / f"selection_{args.budget}_{tag}.json", "w") as f:
        json.dump({
            "selected_cases": selected,
            "method": f"cmaes_{args.fitness}",
            "budget": args.budget,
            "weights": results["best_weights"],
            "weights_normalized": results["best_weights_normalized"],
        }, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
