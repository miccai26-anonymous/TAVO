#!/usr/bin/env python3
"""
BO + Successive Halving meta-selection using nnUNet proxy fitness (PlainConvUNet).

Usage:
    python run_meta_bo_sh.py \
        --budget 250 --generations 20 --eta 3 --n-candidates 50 \
        --R 500 --fitness median_val --best-k 4 --seed 42 \
        --val-cases CASE_A CASE_B CASE_C \
        --datasets DatasetXXX_Name DatasetYYY_Name \
        --preprocessed-dir /path/to/nnUNet_preprocessed \
        --embeddings-dir /path/to/embeddings \
        --output-dir ./outputs/meta/bo_sh_eta3_nnunet500
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
from meta.optimizers.bo_sh import BOSHOptimizer
from meta.fitness.nnunet_proxy import NNUNetProxyFitnessEvaluator
from utils import load_embeddings

DEFAULT_METHODS = ["rds", "less_proxy", "gradmatch_proxy", "kcenter", "diversity", "kmeans", "craig_proxy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--R", type=int, default=500, help="Max steps per candidate")
    parser.add_argument("--eta", type=int, default=3, help="Elimination factor")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
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
    parser.add_argument("--fitness", type=str, default="median_val",
                        choices=["median_val", "best_val", "final_val"])
    parser.add_argument("--best-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"BO + Successive Halving (nnUNet Proxy, PlainConvUNet)")
    print("=" * 60)
    print(f"  Target: {args.target_name}")
    print(f"  R (max steps): {args.R}")
    print(f"  eta: {args.eta}")
    print(f"  Candidates/gen: {args.n_candidates}")
    print(f"  Generations: {args.generations}")
    print(f"  Fitness: {args.fitness} (best_k={args.best_k})")
    print(f"  Budget: {args.budget}")
    print(f"  Crop: {args.crop_size}, Batch: {args.batch_size}")

    methods = [m for m in args.methods if m in SELECTION_METHODS]
    print(f"  Methods: {methods}")

    print("\nLoading embeddings...")
    pool_ids, pool_emb = load_embeddings(args.embeddings_dir / "pool_embeddings.jsonl")
    _, query_emb = load_embeddings(args.embeddings_dir / "query_embeddings.jsonl")
    print(f"  Pool: {len(pool_ids)}")

    print("\nComputing method scores...")
    scores = {}
    for m in methods:
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
        print("  ERROR: No data found!")
        return

    fitness_eval = NNUNetProxyFitnessEvaluator(
        data_dirs=data_dirs, val_cases=args.val_cases,
        batch_size=args.batch_size, lr=args.lr,
        crop_size=args.crop_size, best_k=args.best_k,
    )
    combiner = get_combiner("weighted")(normalize=True)

    optimizer = BOSHOptimizer(methods=methods, seed=args.seed)
    optimizer.initialize({
        "R": args.R, "eta": args.eta, "n_candidates": args.n_candidates,
    })

    eval_log = []

    def fitness_fn(weights, n_steps):
        combined = combiner.combine(scores, weights, methods)
        selected = [c for c, _ in sorted(combined.items(), key=lambda x: -x[1])[:args.budget]]
        t0 = time.time()
        result = fitness_eval.evaluate(selected, n_steps)
        elapsed = time.time() - t0
        eval_log.append({
            "weights": weights.tolist(),
            "weights_dict": {m: float(weights[i]) for i, m in enumerate(methods)},
            "n_steps": n_steps, "fitness": result["fitness"],
            "val_dices": result["val_dices"],
            "final_val_dice": result.get("final_val_dice"),
            "n_train_slices": result.get("n_train_slices"),
            "train_loss_first10": float(np.mean(result["train_losses"][:10])) if result["train_losses"] else None,
            "train_loss_last10": float(np.mean(result["train_losses"][-10:])) if result["train_losses"] else None,
            "elapsed_seconds": elapsed,
        })
        return result["fitness"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.fitness}_eta{args.eta}_{args.seed}"

    def save_checkpoint(gen, best_w, best_f, all_hist, gen_sums, tot_steps):
        best_norm = best_w / best_w.sum() if best_w is not None and best_w.sum() > 0 else best_w
        n_gp_obs = len(optimizer.train_X) if optimizer.train_X is not None else 0
        checkpoint = {
            "config": {
                "optimizer": "bo_sh", "proxy_model": "PlainConvUNet_nnunet2d",
                "target": args.target_name,
                "budget": args.budget, "generations": args.generations, "n_candidates": args.n_candidates,
                "R": args.R, "eta": args.eta, "fitness_type": args.fitness,
                "best_k": args.best_k, "methods": methods, "seed": args.seed,
            },
            "best_weights": {m: float(best_w[i]) for i, m in enumerate(methods)} if best_w is not None else None,
            "best_weights_normalized": {m: float(best_norm[i]) for i, m in enumerate(methods)} if best_norm is not None else None,
            "best_fitness": float(best_f),
            "completed_generations": gen,
            "total_steps": tot_steps,
            "gp_observations": n_gp_obs,
            "generation_summaries": gen_sums,
            "total_time_seconds": time.time() - t_start,
        }
        with open(args.output_dir / f"checkpoint_{tag}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
        with open(args.output_dir / f"history_{tag}.json", "w") as f:
            json.dump(all_hist, f, indent=2)

    print(f"\nRunning {args.generations} generations...")
    total_steps = 0
    all_history = []
    generation_summaries = []
    global_best_fitness = float("-inf")
    global_best_weights = None
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
        global_best_fitness = ckpt["best_fitness"]
        bw = ckpt["best_weights"]
        global_best_weights = np.array([bw[m] for m in methods])
        generation_summaries = ckpt["generation_summaries"]
        all_history = saved_history
        total_steps = ckpt["total_steps"]

        # Rebuild BO state from history (only final-stage survivors)
        import torch
        final_stage = len(optimizer.schedule) - 1
        survivor_entries = [h for h in saved_history if h.get("stage") == final_stage]

        if survivor_entries:
            train_X = torch.tensor([h["weights"] for h in survivor_entries], dtype=torch.float64)
            train_Y = torch.tensor([[h["fitness"]] for h in survivor_entries], dtype=torch.float64)
            optimizer.train_X = train_X
            optimizer.train_Y = train_Y

        optimizer.generation_count = completed_gens
        optimizer.best_fitness = global_best_fitness
        optimizer.best_weights = global_best_weights.copy()

        start_gen = completed_gens
        print(f"  Resumed BO-SH: gen {completed_gens}, best={global_best_fitness:.4f}")

    for gen in range(start_gen, args.generations):
        gen_start = time.time()
        print(f"\n{'='*40} Generation {gen + 1}/{args.generations} {'='*40}")
        eval_log.clear()

        best_w, best_f, sh_history = optimizer.run_generation(fitness_fn)

        for i, h in enumerate(sh_history):
            h["generation"] = gen + 1
            if i < len(eval_log):
                for key in eval_log[i]:
                    if key not in h:
                        h[key] = eval_log[i][key]

        gen_steps = sum(h["steps"] for h in sh_history)
        total_steps += gen_steps
        all_history.extend(sh_history)

        if best_f > global_best_fitness:
            global_best_fitness = best_f
            global_best_weights = best_w.copy()

        gen_fitnesses = [h["fitness"] for h in sh_history]
        stage_counts = {}
        for h in sh_history:
            s = h["stage"]
            stage_counts[s] = stage_counts.get(s, 0) + 1

        n_gp_obs = len(optimizer.train_X) if optimizer.train_X is not None else 0
        sorted_history = sorted(sh_history, key=lambda x: x["fitness"], reverse=True)

        gen_summary = {
            "generation": gen + 1, "best_fitness": best_f,
            "global_best_fitness": global_best_fitness,
            "mean_fitness": float(np.mean(gen_fitnesses)),
            "std_fitness": float(np.std(gen_fitnesses)),
            "min_fitness": float(np.min(gen_fitnesses)),
            "max_fitness": float(np.max(gen_fitnesses)),
            "steps_used": gen_steps, "cumulative_steps": total_steps,
            "candidates_per_stage": stage_counts, "gp_observations": n_gp_obs,
            "best_weights": {m: float(best_w[i]) for i, m in enumerate(methods)},
            "elapsed_seconds": time.time() - gen_start,
            "top_5_rankings": [
                {"rank": rank + 1, "fitness": h["fitness"], "stage": h["stage"],
                 "steps": h["steps"],
                 "weights": {m: h["weights"][i] for i, m in enumerate(methods)},
                 "val_dices": h.get("val_dices", []),
                 "final_val_dice": h.get("final_val_dice")}
                for rank, h in enumerate(sorted_history[:5])
            ],
        }
        generation_summaries.append(gen_summary)

        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (gen + 1)) * (args.generations - gen - 1)
        print(f"  Best: {best_f:.4f} (global: {global_best_fitness:.4f})")
        print(f"  Mean: {gen_summary['mean_fitness']:.4f} +/- {gen_summary['std_fitness']:.4f}")
        print(f"  Steps: {gen_steps}, Cumulative: {total_steps}, GP obs: {n_gp_obs}")
        print(f"  Time: {gen_elapsed:.0f}s, ETA: {eta/60:.0f}min")

        save_checkpoint(gen + 1, global_best_weights, global_best_fitness, all_history, generation_summaries, total_steps)

    # Final results
    best_weights, best_fitness = optimizer.get_best()
    best_norm = best_weights / best_weights.sum() if best_weights.sum() > 0 else best_weights

    print(f"\n{'='*60}")
    print(f"Results: Best fitness: {best_fitness:.4f}")
    for m, w in zip(methods, best_norm):
        print(f"  {m}: {w:.4f}")

    combined = combiner.combine(scores, best_weights, methods)
    sorted_combined = sorted(combined.items(), key=lambda x: -x[1])
    selected = [c for c, _ in sorted_combined[:args.budget]]

    naive = args.generations * args.n_candidates * args.R
    print(f"\nEfficiency: {total_steps} steps (naive: {naive}, speedup: {naive/total_steps:.1f}x)")
    print(f"Total time: {(time.time() - t_start)/3600:.1f}h")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "optimizer": "bo_sh", "proxy_model": "PlainConvUNet_nnunet2d",
            "target": args.target_name, "loss": "DC_and_CE",
            "budget": args.budget, "generations": args.generations, "n_candidates": args.n_candidates,
            "R": args.R, "eta": args.eta, "fitness_type": args.fitness, "best_k": args.best_k,
            "methods": methods, "batch_size": args.batch_size, "lr": args.lr,
            "crop_size": args.crop_size, "seed": args.seed,
        },
        "best_weights": {m: float(best_weights[i]) for i, m in enumerate(methods)},
        "best_weights_normalized": {m: float(best_norm[i]) for i, m in enumerate(methods)},
        "best_fitness": float(best_fitness),
        "total_steps": total_steps, "naive_steps": naive, "speedup": naive / total_steps,
        "total_time_seconds": time.time() - t_start,
        "selected_cases": selected,
        "selected_scores": {c: float(combined[c]) for c in selected},
        "generation_summaries": generation_summaries,
    }

    with open(args.output_dir / f"results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_dir / f"history_{tag}.json", "w") as f:
        json.dump(all_history, f, indent=2)
    with open(args.output_dir / f"selection_{args.budget}_{tag}.json", "w") as f:
        json.dump({
            "selected": selected,
            "selected_scores": {c: float(combined[c]) for c in selected},
            "method": f"bo_sh_nnunet_proxy_{args.fitness}_eta{args.eta}",
            "budget": args.budget, "weights": results["best_weights"],
        }, f, indent=2)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
