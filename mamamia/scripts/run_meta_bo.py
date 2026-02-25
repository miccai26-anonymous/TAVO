#!/usr/bin/env python3
"""
Bayesian Optimization meta-selection using nnUNet proxy fitness (PlainConvUNet).

Usage:
    python run_meta_bo.py \
        --budget 250 --generations 20 --batch-size-bo 8 --n-steps 500 \
        --fitness median_val --best-k 4 --seed 42 \
        --val-cases CASE_A CASE_B CASE_C \
        --datasets DatasetXXX_Name DatasetYYY_Name \
        --preprocessed-dir /path/to/nnUNet_preprocessed \
        --embeddings-dir /path/to/embeddings \
        --output-dir ./outputs/meta/bo_nnunet500
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
from meta import get_combiner
from meta.optimizers.bayesian import BayesianOptimizer
from meta.fitness.nnunet_proxy import NNUNetProxyFitnessEvaluator
from utils import load_embeddings

DEFAULT_METHODS = ["rds", "less_proxy", "gradmatch_proxy", "kcenter", "diversity", "kmeans", "craig_proxy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--batch-size-bo", type=int, default=8, help="Candidates per BO generation")
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
    parser.add_argument("--batch-size", type=int, default=12, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Bayesian Optimization (nnUNet Proxy, PlainConvUNet)")
    print("=" * 60)
    print(f"  Target: {args.target_name}")
    print(f"  Steps per eval: {args.n_steps}")
    print(f"  Fitness: {args.fitness} (best_k={args.best_k})")
    print(f"  Generations: {args.generations}, BO batch: {args.batch_size_bo}")
    print(f"  Budget: {args.budget}")
    print(f"  Crop: {args.crop_size}, Train batch: {args.batch_size}")

    methods = [m for m in args.methods if m in SELECTION_METHODS]
    print(f"  Methods: {methods}")

    print(f"\nLoading embeddings from {args.embeddings_dir}...")
    pool_ids, pool_emb = load_embeddings(args.embeddings_dir / "pool_embeddings.jsonl")
    _, query_emb = load_embeddings(args.embeddings_dir / "query_embeddings.jsonl")
    print(f"  Pool: {len(pool_ids)}, Query: {query_emb.shape[0]}")

    print(f"\nComputing scores for {len(methods)} methods...")
    scores = {}
    for m in methods:
        print(f"  Computing scores for {m}...")
        method = get_selection_method(m)(seed=args.seed)
        result = method.select(pool_ids, args.budget, embeddings=pool_emb, query_embeddings=query_emb)
        scores[m] = result.scores

    print(f"\nSetting up nnUNet proxy fitness evaluator...")
    data_dirs = []
    for ds in args.datasets:
        p = Path(args.preprocessed_dir) / ds / "nnUNetPlans_2d"
        if p.exists():
            data_dirs.append(p)
            print(f"    Found: {ds}")
    if not data_dirs:
        print("  ERROR: No data directories found!")
        return

    fitness_eval = NNUNetProxyFitnessEvaluator(
        data_dirs=data_dirs, val_cases=args.val_cases,
        batch_size=args.batch_size, lr=args.lr,
        crop_size=args.crop_size, best_k=args.best_k,
    )
    combiner = get_combiner("weighted")(normalize=True)

    optimizer = BayesianOptimizer(methods=methods, seed=args.seed)
    optimizer.initialize({"batch_size": args.batch_size_bo})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.fitness}_{args.seed}"

    def save_checkpoint(gen, all_hist, gen_sums):
        bw, bf = optimizer.get_best()
        bn = bw / bw.sum() if bw.sum() > 0 else bw
        checkpoint = {
            "config": {
                "optimizer": "bayesian", "proxy_model": "PlainConvUNet_nnunet2d",
                "target": args.target_name,
                "budget": args.budget, "generations": args.generations,
                "batch_size_bo": args.batch_size_bo, "methods": methods,
                "n_steps": args.n_steps, "fitness_type": args.fitness,
                "best_k": args.best_k, "seed": args.seed,
            },
            "best_weights": {m: float(bw[i]) for i, m in enumerate(methods)},
            "best_weights_normalized": {m: float(bn[i]) for i, m in enumerate(methods)},
            "best_fitness": float(bf),
            "completed_generations": gen,
            "total_evaluations": len(all_hist),
            "generation_summaries": gen_sums,
            "total_time_seconds": time.time() - t_start,
        }
        with open(args.output_dir / f"checkpoint_{tag}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
        with open(args.output_dir / f"history_{tag}.json", "w") as f:
            json.dump(all_hist, f, indent=2)

    print(f"\nRunning {args.generations} generations...")
    all_history = []
    generation_summaries = []
    total_evals = 0
    t_start = time.time()
    start_gen = 0

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
        all_history = saved_history
        generation_summaries = ckpt["generation_summaries"]
        total_evals = len(saved_history)

        # Rebuild BO state from history
        import torch
        train_X = torch.tensor([h["weights"] for h in saved_history], dtype=torch.float64)
        train_Y = torch.tensor([[h["fitness"]] for h in saved_history], dtype=torch.float64)
        optimizer.train_X = train_X
        optimizer.train_Y = train_Y
        optimizer.generation_count = completed_gens

        best_idx = train_Y.argmax().item()
        optimizer.best_fitness = train_Y[best_idx].item()
        optimizer.best_weights = train_X[best_idx].numpy()

        start_gen = completed_gens
        print(f"  Resumed BO: gen {completed_gens}, {total_evals} obs, best={optimizer.best_fitness:.4f}")

    for gen in range(start_gen, args.generations):
        gen_start = time.time()
        print(f"\n{'='*40} Generation {gen + 1}/{args.generations} {'='*40}")

        candidates = optimizer.ask_batch()
        gen_records = []

        for i, weights in enumerate(candidates):
            total_evals += 1
            combined = combiner.combine(scores, weights, methods)
            selected = [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])[:args.budget]]

            t0 = time.time()
            result = fitness_eval.evaluate(selected, args.n_steps)
            fitness = result["fitness"]
            elapsed = time.time() - t0

            eval_record = {
                "generation": gen + 1, "eval_in_generation": i + 1,
                "global_eval_idx": total_evals,
                "weights": weights.tolist(),
                "weights_dict": {m: float(weights[j]) for j, m in enumerate(methods)},
                "fitness": float(fitness),
                "val_dices": result["val_dices"],
                "train_losses_summary": {
                    "first_10_mean": float(np.mean(result["train_losses"][:10])) if result["train_losses"] else None,
                    "last_10_mean": float(np.mean(result["train_losses"][-10:])) if result["train_losses"] else None,
                },
                "final_val_dice": result.get("final_val_dice"),
                "n_train_slices": result.get("n_train_slices"),
                "elapsed_seconds": elapsed,
            }
            gen_records.append(eval_record)
            all_history.append(eval_record)

            total_elapsed = time.time() - t_start
            total_expected = args.generations * args.batch_size_bo
            eta = (total_elapsed / total_evals) * (total_expected - total_evals)
            print(f"  [{total_evals}/{total_expected}] Gen {gen+1}.{i+1}: "
                  f"fitness={fitness:.4f}, best={optimizer.best_fitness:.4f} "
                  f"[{elapsed:.0f}s, ETA {eta/60:.0f}min]")

        fitnesses = [r["fitness"] for r in gen_records]
        optimizer.tell_batch(candidates, fitnesses)

        gen_fitnesses = [r["fitness"] for r in gen_records]
        sorted_gen = sorted(gen_records, key=lambda x: x["fitness"], reverse=True)
        gen_summary = {
            "generation": gen + 1,
            "best_fitness_this_gen": float(max(gen_fitnesses)),
            "global_best_fitness": float(optimizer.best_fitness),
            "mean_fitness": float(np.mean(gen_fitnesses)),
            "std_fitness": float(np.std(gen_fitnesses)),
            "best_weights_this_gen": sorted_gen[0]["weights_dict"],
            "rankings": [
                {"rank": i + 1, "fitness": e["fitness"], "weights": e["weights_dict"],
                 "val_dices": e["val_dices"], "final_val_dice": e["final_val_dice"]}
                for i, e in enumerate(sorted_gen)
            ],
            "elapsed_seconds": time.time() - gen_start,
        }
        generation_summaries.append(gen_summary)

        print(f"  --- Gen {gen+1} summary: best={max(gen_fitnesses):.4f}, "
              f"mean={np.mean(gen_fitnesses):.4f} +/- {np.std(gen_fitnesses):.4f}, "
              f"global_best={optimizer.best_fitness:.4f}")
        save_checkpoint(gen + 1, all_history, generation_summaries)

    # Final results
    best_weights, best_fitness = optimizer.get_best()
    best_norm = best_weights / best_weights.sum() if best_weights.sum() > 0 else best_weights

    print(f"\n{'='*60}")
    print(f"Optimization complete! Best fitness: {best_fitness:.4f}")
    for m, w in zip(methods, best_norm):
        print(f"  {m}: {w:.4f}")

    combined = combiner.combine(scores, best_weights, methods)
    sorted_cases = sorted(combined.items(), key=lambda x: -x[1])
    selected_cases = [c for c, _ in sorted_cases[:args.budget]]

    results = {
        "config": {
            "optimizer": "bayesian", "proxy_model": "PlainConvUNet_nnunet2d",
            "target": args.target_name, "loss": "DC_and_CE",
            "budget": args.budget, "generations": args.generations,
            "batch_size_bo": args.batch_size_bo, "methods": methods,
            "n_steps": args.n_steps, "fitness_type": args.fitness,
            "best_k": args.best_k, "batch_size": args.batch_size,
            "lr": args.lr, "crop_size": args.crop_size, "seed": args.seed,
        },
        "best_weights": {m: float(best_weights[i]) for i, m in enumerate(methods)},
        "best_weights_normalized": {m: float(best_norm[i]) for i, m in enumerate(methods)},
        "best_fitness": float(best_fitness),
        "total_evaluations": total_evals,
        "total_time_seconds": time.time() - t_start,
        "selected_cases": selected_cases,
        "selected_scores": {c: float(combined[c]) for c in selected_cases},
        "generation_summaries": generation_summaries,
    }

    with open(args.output_dir / f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_dir / f"history_{tag}.json", "w") as f:
        json.dump(all_history, f, indent=2)
    with open(args.output_dir / f"selection_{args.budget}_{tag}.json", "w") as f:
        json.dump({
            "selected": selected_cases,
            "selected_scores": {c: float(combined[c]) for c in selected_cases},
            "method": f"bo_nnunet_proxy_{args.fitness}",
            "budget": args.budget,
            "weights": results["best_weights"],
            "weights_normalized": results["best_weights_normalized"],
        }, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
