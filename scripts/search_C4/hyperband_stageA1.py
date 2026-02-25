#!/usr/bin/env python3
import json, shutil
from pathlib import Path
import numpy as np
import torch  # for SobolEngine

from scripts.search_C4.utils_cma import (
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
TEMPLATE_YAML = PROJECT_ROOT / "configs_C4_cma/template.yaml"

SCORE_ROOT = PROJECT_ROOT / "data/splits_C4_mix_scores"
SPLIT_ROOT = PROJECT_ROOT / "data/splits_C4_stageA1"
OUT_ROOT   = PROJECT_ROOT / "outputs_C4_stageA1"

T = 50  # target size per T

# ===============================
# Warmup checkpoints (stable epoch = 7)
# ===============================
WARMUP_CKPT_MAP = {
    "repeat01": "/path/to/workspace/"
                "ANON_USER/EfficientVit/outputs_C4_new/"
                "source_plus_target_repeat01/epoch_007.pt",
    "repeat02": "/path/to/workspace/"
                "ANON_USER/EfficientVit/outputs_C4_new/"
                "source_plus_target_repeat02/epoch_007.pt",
    "repeat03": "/path/to/workspace/"
                "ANON_USER/EfficientVit/outputs_C4_new/"
                "source_plus_target_repeat03/epoch_007.pt",
    "repeat04": "/path/to/workspace/"
                "ANON_USER/EfficientVit/outputs_C4_new/"
                "source_plus_target_repeat04/epoch_007.pt",
    "repeat05": "/path/to/workspace/"
                "ANON_USER/EfficientVit/outputs_C4_new/"
                "source_plus_target_repeat05/epoch_007.pt",
}

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
# One candidate run (resume-aware)
# =====================================================
def run_candidate(
    repeat_id,
    budget_T,
    run_tag,
    w_rds,
    w_less,
    max_iters,
    seeds,
    warmup_ckpt=None, 
    # resume_ckpt=None,
):
    budget = budget_T * T
    tag = f"wR{w_rds:.3f}_wL{w_less:.3f}_it{max_iters}"

    split_dir = SPLIT_ROOT / f"repeat{repeat_id:02d}" / f"{budget_T}T" / run_tag / tag
    out_dir   = OUT_ROOT   / f"repeat{repeat_id:02d}" / f"{budget_T}T" / run_tag / tag
    yaml_path = out_dir / "train_config.yaml"
    train_txt = split_dir / "train_subjects.txt"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    rds, less = load_norm_scores(SCORE_ROOT, repeat_id)
    build_subset(rds, less, w_rds, w_less, budget, train_txt)
    
    # -----------------------------
    # Decide initialization checkpoint
    # -----------------------------
    
    init_ckpt = warmup_ckpt
    ckpt_mode = "init_from_warmup"

    print(f"🔥 [A1] CKPT mode = {ckpt_mode}")
    print(f"🔥 [A1] Using checkpoint = {init_ckpt}")

    build_yaml(
        TEMPLATE_YAML,
        yaml_path,
        train_txt,
        out_dir,
        max_iters=max_iters,
        resume_ckpt=init_ckpt,
        stage="A",
    )

    run_training(yaml_path, seeds=seeds)

    # tr, va = load_losses(out_dir)
    # fit, info = compute_fitness(tr, va)
    
    dice = compute_fast_dice(
        out_dir,
        max_subjects=5,   # Stage A1
    )

    fit = dice

    print(
        f"[A1][fast-dice] "
        f"wR={w_rds:.3f} wL={w_less:.3f} | "
        f"iters={max_iters} | "
        f"dice={dice:.4f}"
    )

    ckpt = out_dir / "latest.pt"

    if ckpt.exists():
        ckpt.unlink()

    return {
        "w": (w_rds, w_less),
        "fitness": float(fit),
        "iters": int(max_iters),
        "ckpt": str(ckpt),
        "out_dir": str(out_dir),
    }

# =====================================================
# Stage-A1: Hyperband / Successive Halving
# =====================================================
def hyperband_stageA(repeat_id, budget_T, run_tag, seeds="0", seed=0):
    iters = HB_CFG["iters"]
    n_keep = HB_CFG["n_keep"]
    n_sobol = HB_CFG["n_sobol_round0"]
    warmup_ckpt = WARMUP_CKPT_MAP[f"repeat{repeat_id:02d}"]
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
                repeat_id=repeat_id,
                budget_T=budget_T,
                run_tag=run_tag,
                w_rds=w_rds,
                w_less=w_less,
                max_iters=max_it,
                seeds=seeds,
                warmup_ckpt=warmup_ckpt,
                # resume_ckpt=ckpt,
            )

            round_results.append(res)
            records.append(res)

        round_results.sort(key=lambda x: x["fitness"], reverse=True)
        survivors = round_results[:keep_k]

        pool = [r["w"] for r in survivors]
        # prev_ckpt = {r["w"]: r["ckpt"] for r in survivors}

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
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--budget_T", type=int, required=True)
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = hyperband_stageA(
        repeat_id=args.repeat,
        budget_T=args.budget_T,
        run_tag=args.run_tag,
        seeds=args.seeds,
        seed=args.seed,
    )

    save_dir = OUT_ROOT / f"repeat{args.repeat:02d}" / f"{args.budget_T}T" / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "stageA1_hyperband.json").write_text(json.dumps(out, indent=2))

    print("\n✅ Final candidates:")
    for w in out["final_candidates"]:
        print(w)
