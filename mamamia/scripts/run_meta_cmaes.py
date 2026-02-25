#!/usr/bin/env python3
"""
CMA-ES meta-optimization using nnUNet proxy fitness (PlainConvUNet, configurable steps).

Usage:
    python run_meta_cmaes.py \
        --budget 250 --generations 20 --popsize 8 --n-steps 500 \
        --fitness median_val --best-k 4 --seed 42 \
        --val-cases CASE_A CASE_B CASE_C \
        --datasets DatasetXXX_Name DatasetYYY_Name \
        --preprocessed-dir /path/to/nnUNet_preprocessed \
        --embeddings-dir /path/to/embeddings \
        --output-dir ./outputs/meta/cmaes_nnunet500
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np

# Project root: mamamia/
MAMAMIA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAMAMIA_DIR))

from selection import SELECTION_METHODS, get_selection_method
from meta import get_optimizer, get_combiner
from meta.fitness.nnunet_proxy import NNUNetProxyFitnessEvaluator
from utils import load_embeddings

DEFAULT_METHODS = ["rds", "less_proxy", "gradmatch_proxy", "kcenter", "diversity", "kmeans", "craig_proxy"]


def compute_all_scores(methods, pool_ids, pool_embeddings, query_embeddings, budget, seed=42):
    all_scores = {}
    for method_name in methods:
        print(f"  Computing scores for {method_name}...")
        method_cls = get_selection_method(method_name)
        method = method_cls(seed=seed)
        result = method.select(pool_ids=pool_ids, budget=budget,
                               embeddings=pool_embeddings, query_embeddings=query_embeddings)
        all_scores[method_name] = result.scores
    return all_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Directory containing pool_embeddings.jsonl and query_embeddings.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for results, checkpoints, and history")
    parser.add_argument("--preprocessed-dir", type=Path, required=True,
                        help="Path to nnUNet_preprocessed directory")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="Dataset folder names under preprocessed-dir (e.g. DatasetXXX_Name)")
    parser.add_argument("--val-cases", type=str, nargs="+", required=True,
                        help="Validation case IDs for fitness evaluation")
    parser.add_argument("--target-name", type=str, default="Target",
                        help="Name of the target domain (for logging and JSON metadata)")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--fitness", type=str, default="median_val",
                        choices=["median_val", "best_val", "final_val"])
    parser.add_argument("--best-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"CMA-ES Meta-Optimizer (nnUNet Proxy, PlainConvUNet)")
    print("=" * 60)
    print(f"  Target: {args.target_name}")
    print(f"  Steps per eval: {args.n_steps}")
    print(f"  Fitness: {args.fitness} (best_k={args.best_k})")
    print(f"  Generations: {args.generations}, Population: {args.popsize}")
    print(f"  Budget: {args.budget}")
    print(f"  Crop size: {args.crop_size}, Batch size: {args.batch_size}")

    methods = [m for m in args.methods if m in SELECTION_METHODS]
    if len(methods) < len(args.methods):
        print(f"  Warning: Missing methods: {set(args.methods) - set(methods)}")

    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings_dir}...")
    pool_ids, pool_emb = load_embeddings(args.embeddings_dir / "pool_embeddings.jsonl")
    _, query_emb = load_embeddings(args.embeddings_dir / "query_embeddings.jsonl")
    print(f"  Pool: {len(pool_ids)}, Query: {query_emb.shape[0]}")

    # Compute selection scores
    print(f"\nComputing scores for {len(methods)} methods...")
    scores = compute_all_scores(methods, pool_ids, pool_emb, query_emb, args.budget, args.seed)

    # Setup fitness evaluator
    print(f"\nSetting up nnUNet proxy fitness evaluator...")
    data_dirs = []
    for ds in args.datasets:
        p = Path(args.preprocessed_dir) / ds / "nnUNetPlans_2d"
        if p.exists():
            data_dirs.append(p)
            print(f"    Found: {ds}")
    if not data_dirs:
        print("  ERROR: No data directories found! Exiting.")
        return

    fitness_eval = NNUNetProxyFitnessEvaluator(
        data_dirs=data_dirs, val_cases=args.val_cases,
        batch_size=args.batch_size, lr=args.lr,
        crop_size=args.crop_size, best_k=args.best_k,
    )

    # Initialize CMA-ES
    combiner = get_combiner("weighted")(normalize=True)
    optimizer_cls = get_optimizer("cmaes")
    optimizer = optimizer_cls(methods=methods, seed=args.seed)
    optimizer.initialize({
        "sigma0": 0.3,
        "maxiter": args.generations,
        "popsize": args.popsize,
        "bounds": [0, 1],
    })

    # Corner evaluations (pure methods)
    print(f"\nEvaluating corner cases (pure methods)...")
    corner_results = {}
    for i, method in enumerate(methods):
        t0 = time.time()
        corner_w = np.zeros(len(methods))
        corner_w[i] = 1.0
        combined = combiner.combine(scores, corner_w, methods)
        selected = [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])[:args.budget]]

        result = fitness_eval.evaluate(selected, args.n_steps)
        corner_results[method] = result
        elapsed = time.time() - t0
        print(f"    {method}: fitness={result['fitness']:.4f} "
              f"(final_dice={result.get('final_val_dice', 'N/A')}) [{elapsed:.0f}s]")

    # Checkpoint helper
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.fitness}_{args.seed}"

    def save_checkpoint(gen, best_weights, best_fitness, all_history, generation_summaries):
        best_w_norm = best_weights / best_weights.sum() if best_weights is not None and best_weights.sum() > 0 else best_weights
        checkpoint = {
            "config": {
                "optimizer": "cmaes", "proxy_model": "PlainConvUNet_nnunet2d",
                "target": args.target_name,
                "budget": args.budget, "generations": args.generations, "popsize": args.popsize,
                "methods": methods, "n_steps": args.n_steps, "fitness_type": args.fitness,
                "best_k": args.best_k, "seed": args.seed,
            },
            "best_weights": {m: float(best_weights[i]) for i, m in enumerate(methods)} if best_weights is not None else None,
            "best_weights_normalized": {m: float(best_w_norm[i]) for i, m in enumerate(methods)} if best_w_norm is not None else None,
            "best_fitness": float(best_fitness),
            "completed_generations": gen,
            "generation_summaries": generation_summaries,
            "total_time_seconds": time.time() - t_start,
        }
        with open(args.output_dir / f"checkpoint_{tag}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
        with open(args.output_dir / f"history_{tag}.json", "w") as f:
            json.dump(all_history, f, indent=2)

    # Main optimization loop
    print(f"\nRunning CMA-ES optimization ({args.generations} generations)...")
    best_weights = None
    best_fitness = float("-inf")
    all_history = []
    generation_summaries = []
    total_evals = args.generations * args.popsize
    current_gen_evals = []
    global_eval_idx = 0
    t_start = time.time()
    start_eval = 0

    # === Resume from checkpoint ===
    checkpoint_path = args.output_dir / f"checkpoint_{tag}.json"
    history_path = args.output_dir / f"history_{tag}.json"

    if checkpoint_path.exists() and history_path.exists():
        print(f"\n>>> Resuming from checkpoint...")
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        with open(history_path) as f:
            saved_history = json.load(f)

        completed_gens = ckpt["completed_generations"]
        bw = ckpt["best_weights"]
        best_weights = np.array([bw[m] for m in methods])
        best_fitness = ckpt["best_fitness"]
        generation_summaries = ckpt["generation_summaries"]
        all_history = saved_history

        # Replay through optimizer to restore CMA-ES internal state
        print(f"  Replaying {len(saved_history)} evaluations...")
        for record in saved_history:
            _ = optimizer.ask()  # advance CMA-ES internal state
            optimizer.tell(np.array(record["weights"]), record["fitness"])

        start_eval = completed_gens * args.popsize
        global_eval_idx = len(saved_history)
        print(f"  Resumed: gen {completed_gens}, {global_eval_idx} evals, best={best_fitness:.4f}")

    for eval_idx in range(start_eval, total_evals):
        weights = optimizer.ask()
        gen = eval_idx // args.popsize + 1
        eval_in_gen = eval_idx % args.popsize + 1
        global_eval_idx += 1

        combined = combiner.combine(scores, weights, methods)
        selected = [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])[:args.budget]]

        t0 = time.time()
        result = fitness_eval.evaluate(selected, args.n_steps)
        fitness = result["fitness"]
        elapsed = time.time() - t0

        optimizer.tell(weights, fitness)

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = weights.copy()

        eval_record = {
            "generation": gen, "eval_in_generation": eval_in_gen,
            "global_eval_idx": global_eval_idx,
            "weights": weights.tolist(),
            "weights_dict": {m: float(weights[i]) for i, m in enumerate(methods)},
            "fitness": float(fitness),
            "val_dices": result["val_dices"],
            "train_losses_summary": {
                "first_10_mean": float(np.mean(result["train_losses"][:10])) if result["train_losses"] else None,
                "last_10_mean": float(np.mean(result["train_losses"][-10:])) if result["train_losses"] else None,
                "overall_mean": float(np.mean(result["train_losses"])) if result["train_losses"] else None,
            },
            "final_val_dice": result.get("final_val_dice"),
            "n_train_slices": result.get("n_train_slices"),
            "elapsed_seconds": elapsed,
        }
        all_history.append(eval_record)
        current_gen_evals.append(eval_record)

        total_elapsed = time.time() - t_start
        eta = (total_elapsed / global_eval_idx) * (total_evals - global_eval_idx)
        print(f"  [{global_eval_idx}/{total_evals}] Gen {gen}.{eval_in_gen}: "
              f"fitness={fitness:.4f}, global_best={best_fitness:.4f} "
              f"[{elapsed:.0f}s, ETA {eta/60:.0f}min]")

        if eval_in_gen == args.popsize:
            gen_fitnesses = [e["fitness"] for e in current_gen_evals]
            sorted_gen = sorted(current_gen_evals, key=lambda x: x["fitness"], reverse=True)
            gen_summary = {
                "generation": gen,
                "best_fitness_this_gen": float(max(gen_fitnesses)),
                "global_best_fitness": float(best_fitness),
                "mean_fitness": float(np.mean(gen_fitnesses)),
                "std_fitness": float(np.std(gen_fitnesses)),
                "min_fitness": float(min(gen_fitnesses)),
                "max_fitness": float(max(gen_fitnesses)),
                "best_weights_this_gen": sorted_gen[0]["weights_dict"],
                "rankings": [
                    {"rank": i + 1, "fitness": e["fitness"], "weights": e["weights_dict"],
                     "val_dices": e["val_dices"], "final_val_dice": e["final_val_dice"]}
                    for i, e in enumerate(sorted_gen)
                ],
            }
            generation_summaries.append(gen_summary)
            print(f"  --- Gen {gen} summary: best={max(gen_fitnesses):.4f}, "
                  f"mean={np.mean(gen_fitnesses):.4f} +/- {np.std(gen_fitnesses):.4f}, "
                  f"global_best={best_fitness:.4f}")
            save_checkpoint(gen, best_weights, best_fitness, all_history, generation_summaries)
            current_gen_evals = []

    # Final results
    best_w_norm = best_weights / best_weights.sum() if best_weights.sum() > 0 else best_weights
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best weights (normalized):")
    for m, w in zip(methods, best_w_norm):
        print(f"  {m}: {w:.4f}")

    combined = combiner.combine(scores, best_weights, methods)
    sorted_cases = sorted(combined.items(), key=lambda x: -x[1])
    selected_cases = [c for c, _ in sorted_cases[:args.budget]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "optimizer": "cmaes", "proxy_model": "PlainConvUNet_nnunet2d",
            "target": args.target_name, "loss": "DC_and_CE",
            "budget": args.budget, "generations": args.generations, "popsize": args.popsize,
            "methods": methods, "n_steps": args.n_steps, "fitness_type": args.fitness,
            "best_k": args.best_k, "batch_size": args.batch_size, "lr": args.lr,
            "crop_size": args.crop_size, "seed": args.seed,
        },
        "best_weights": {m: float(best_weights[i]) for i, m in enumerate(methods)},
        "best_weights_normalized": {m: float(best_w_norm[i]) for i, m in enumerate(methods)},
        "best_fitness": float(best_fitness),
        "corner_fitnesses": {m: float(r["fitness"]) for m, r in corner_results.items()},
        "corner_details": {
            m: {"fitness": r["fitness"], "val_dices": r["val_dices"], "final_val_dice": r["final_val_dice"]}
            for m, r in corner_results.items()
        },
        "selected_cases": selected_cases,
        "selected_scores": {c: float(combined[c]) for c in selected_cases},
        "generation_summaries": generation_summaries,
        "total_time_seconds": time.time() - t_start,
    }

    tag = f"{args.fitness}_{args.seed}"
    with open(args.output_dir / f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_dir / f"history_{tag}.json", "w") as f:
        json.dump(all_history, f, indent=2)
    with open(args.output_dir / f"selection_{args.budget}_{tag}.json", "w") as f:
        json.dump({
            "selected": selected_cases,
            "selected_scores": {c: float(combined[c]) for c in selected_cases},
            "method": f"cmaes_nnunet_proxy_{args.fitness}",
            "budget": args.budget,
            "weights": results["best_weights"],
            "weights_normalized": results["best_weights_normalized"],
        }, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
