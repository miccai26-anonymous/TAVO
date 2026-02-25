#!/usr/bin/env python3
import json, shutil
from pathlib import Path
import numpy as np
import torch  # for SobolEngine

from scripts.search_TCGA.utils_cma import (
    load_norm_scores,
    build_subset,
    build_yaml,
    run_training,
    compute_fast_dice,
    # load_losses,
    # compute_fitness,
)

# =====================================================
# Paths
# =====================================================
PROJECT_ROOT = Path("/path/to/project")
TEMPLATE_YAML = PROJECT_ROOT / "configs_TCGA_cma/template.yaml"

SCORE_ROOT = PROJECT_ROOT / "data/splits_TCGA_mix_scores"
SPLIT_ROOT = PROJECT_ROOT / "data/splits_TCGA_stageA1"
OUT_ROOT   = PROJECT_ROOT / "outputs_TCGA_stageA1"

T = 50  # target size per T

# ===============================
# Warmup checkpoints (stable epoch = 7)
# ===============================
WARMUP_CKPT = (
    "/path/to/workspace/"
    "ANON_USER/EfficientVit/outputs_TCGA/"
    "baseline3_brats21_source_plus_target/epoch_007.pt"
)

# =====================================================
# Hyperband / SH config
# =====================================================
HB_CFG = {
    "iters": [200, 500, 1000],      # successive budgets
    "n_sobol_round0": 18,           # ⭐ Sobol candidates in round0
    "n_keep": [10, 6, 3],           # survivors per round
}

# =====================================================
# Sobol sampling (corners only when specified)
# =====================================================
def sample_weights_sobol(n_random: int, seed: int = 0, include_corners: bool = False):
    ws = []

    if include_corners:
        ws.append((1.0, 0.0))
        ws.append((0.0, 1.0))

    if n_random > 0:
        eng = torch.quasirandom.SobolEngine(
            dimension=1, scramble=True, seed=seed
        )
        x = eng.draw(n_random).squeeze(1).cpu().numpy()

        eps = 1e-6
        x = np.clip(x, eps, 1.0 - eps)

        for w in x:
            ws.append((float(w), float(1.0 - w)))

    # deduplicate
    uniq, seen = [], set()
    for wr, wl in ws:
        key = (round(wr, 6), round(wl, 6))
        if key not in seen:
            seen.add(key)
            uniq.append((wr, wl))
    return uniq

# =====================================================
# standardize seeds input
# =====================================================
def parse_seeds(seeds):
    """
    Accepts: "0,1,2" or "0 1 2" or [0,1,2] or "0"
    Returns: [0,1,2]
    """
    if isinstance(seeds, (list, tuple)):
        return [int(s) for s in seeds]
    if isinstance(seeds, str):
        s = seeds.replace(",", " ").split()
        return [int(x) for x in s]
    return [int(seeds)]


def aggregate_fitness(dice_list, mode="mean"):
    dice_list = [float(x) for x in dice_list]
    if len(dice_list) == 0:
        return 0.0
    if mode == "median":
        return float(np.median(dice_list))
    return float(np.mean(dice_list))

# =====================================================
# One candidate run (resume-aware)
# =====================================================
def run_candidate(
    budget_T,
    run_tag,
    w_rds,
    w_less,
    max_iters,
    seeds,
    warmup_ckpt=None,
    # resume_ckpt=None,      # now can be dict: {seed(int): ckpt_path(str)}
    fitness_agg="median",    # "mean" or "median"
):
    budget = budget_T * T
    tag = f"wR{w_rds:.3f}_wL{w_less:.3f}_it{max_iters}"

    split_dir = SPLIT_ROOT / f"{budget_T}T" / run_tag / tag
    out_dir   = OUT_ROOT   / f"{budget_T}T" / run_tag / tag
    train_txt = split_dir / "train_subjects.txt"

    rds, less = load_norm_scores(SCORE_ROOT, repeat_id=None)
    build_subset(rds, less, w_rds, w_less, budget, train_txt)

    seeds_list = parse_seeds(seeds)
    out_dir.mkdir(parents=True, exist_ok=True)

    dice_per_seed = {}
    ckpt_per_seed = {}
    out_dir_per_seed = {}

    for sd in seeds_list:
        seed_out = out_dir / f"seed{sd}"
        seed_yaml = seed_out / "train_config.yaml"

        if seed_out.exists():
            shutil.rmtree(seed_out)
        seed_out.mkdir(parents=True, exist_ok=True)

        # -------- init ckpt (per-seed) --------
        seed_resume = None
        # if isinstance(resume_ckpt, dict):
        #     seed_resume = resume_ckpt.get(sd, None)
        # elif isinstance(resume_ckpt, str):
        #     seed_resume = resume_ckpt  # backward-compat (not recommended)

        if seed_resume is not None:
            init_ckpt = seed_resume
            ckpt_mode = "resume_from_previous_round"
        else:
            init_ckpt = warmup_ckpt
            ckpt_mode = "init_from_warmup"

        print(f"🔥 [A1][seed{sd}] CKPT mode = {ckpt_mode}")
        print(f"🔥 [A1][seed{sd}] Using checkpoint = {init_ckpt}")

        build_yaml(
            TEMPLATE_YAML,
            seed_yaml,
            train_txt,
            seed_out,
            max_iters=max_iters,
            resume_ckpt=init_ckpt,
            stage="A",
        )

        # IMPORTANT: run one seed at a time to avoid overwriting
        run_training(seed_yaml, seeds=str(sd))

        dice = compute_fast_dice(
            seed_out,
            max_subjects=20,   # Stage A1 (TCGA)
        )

        dice_per_seed[sd] = float(dice)
        ckpt_path = seed_out / "latest.pt"
        ckpt_per_seed[sd] = str(ckpt_path)
        out_dir_per_seed[sd] = str(seed_out)

        print(
            f"[A1][fast-dice][seed{sd}] "
            f"wR={w_rds:.3f} wL={w_less:.3f} | "
            f"iters={max_iters} | dice={dice:.4f}"
        )

    fit = aggregate_fitness(list(dice_per_seed.values()), mode=fitness_agg)

    dice_vals = list(dice_per_seed.values())
    mean_d = float(np.mean(dice_vals))
    median_d = float(np.median(dice_vals))
    std_d = float(np.std(dice_vals))

    print("\n" + "-" * 72)
    print(
        f"[A1][SUMMARY] wR={w_rds:.3f} wL={w_less:.3f} | iters={max_iters}"
    )
    for sd in sorted(dice_per_seed):
        print(f"   seed{sd}: dice = {dice_per_seed[sd]:.4f}")
    print(
        f"   ▶ mean   = {mean_d:.4f}\n"
        f"   ▶ median = {median_d:.4f}\n"
        f"   ▶ std    = {std_d:.4f}\n"
        f"   ▶ fitness ({fitness_agg}) = "
        f"{median_d if fitness_agg=='median' else mean_d:.4f}"
    )
    print("-" * 72 + "\n")


    return {
        "w": (w_rds, w_less),
        "fitness": float(fit),
        "iters": int(max_iters),
        "dice_per_seed": dice_per_seed,
        "ckpt_per_seed": ckpt_per_seed,
        "out_dir_per_seed": out_dir_per_seed,
        "out_dir": str(out_dir),
    }


# =====================================================
# Stage-A1: Hyperband / Successive Halving
# =====================================================
def hyperband_stageA(budget_T, run_tag, seeds="0", seed=0):
    iters = HB_CFG["iters"]
    n_keep = HB_CFG["n_keep"]
    n_sobol = HB_CFG["n_sobol_round0"]
    warmup_ckpt = WARMUP_CKPT
    assert warmup_ckpt is not None
    assert Path(warmup_ckpt).exists(), f"Warmup ckpt not found: {warmup_ckpt}"

    assert len(iters) == len(n_keep)

    records = []
    # prev_ckpt = {}

    # -----------------------------
    # Round0: Sobol + corners
    # -----------------------------
    pool = sample_weights_sobol(
        n_random=n_sobol,
        seed=seed,
        include_corners=True,
    )
    print(f"[Round0] eval = {len(pool)} = {n_sobol} Sobol + 2 corners")

    for ridx, (keep_k, max_it) in enumerate(zip(n_keep, iters)):
        print(f"\n🚀 Round {ridx} | iters={max_it} | eval={len(pool)} | keep={keep_k}")

        round_results = []

        for (w_rds, w_less) in pool:
            # ckpt = prev_ckpt.get((w_rds, w_less), None)

            res = run_candidate(
                budget_T=budget_T,
                run_tag=run_tag,
                w_rds=w_rds,
                w_less=w_less,
                max_iters=max_it,
                seeds=seeds,
                warmup_ckpt=warmup_ckpt,
                # resume_ckpt=ckpt,
                fitness_agg="median",
            )

            round_results.append(res)
            records.append(res)

        round_results.sort(key=lambda x: x["fitness"], reverse=True)
        survivors = round_results[:keep_k]

        print("\n📊 [A1][Round Ranking]")
        for i, r in enumerate(round_results[:min(keep_k, 10)]):
            wR, wL = r["w"]
            print(
                f"  #{i+1:02d}  wR={wR:.3f} wL={wL:.3f} | "
                f"fitness={r['fitness']:.4f}"
            )

        pool = [r["w"] for r in survivors]
        # prev_ckpt = {r["w"]: r["ckpt_per_seed"] for r in survivors}

        if len(pool) <= 1:
            break

    return {
        "all_records": records,
        "final_candidates": pool,
    }

# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_T", type=int, required=True)
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = hyperband_stageA(
        budget_T=args.budget_T,
        run_tag=args.run_tag,
        seeds=args.seeds,
        seed=args.seed,
    )

    save_dir = OUT_ROOT / f"{args.budget_T}T" / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "stageA1_hyperband.json").write_text(json.dumps(out, indent=2))

    print("\n✅ Final candidates:")
    for w in out["final_candidates"]:
        print(w)
