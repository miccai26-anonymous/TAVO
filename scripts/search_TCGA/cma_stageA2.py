#!/usr/bin/env python3
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np

from scripts.search_TCGA.utils_cma import (
    load_norm_scores,
    build_subset,
    build_yaml,
    run_training,
    compute_fast_dice,
)

# =====================================================
# Paths (TCGA)
# =====================================================
PROJECT_ROOT = Path("/path/to/project")
TEMPLATE_YAML = PROJECT_ROOT / "configs_TCGA_cma/template.yaml"

SCORE_ROOT = PROJECT_ROOT / "data/splits_TCGA_mix_scores"
SPLIT_ROOT = PROJECT_ROOT / "data/splits_TCGA_stageA2"
OUT_ROOT   = PROJECT_ROOT / "outputs_TCGA_stageA2"

T = 50

# ===============================
# Warmup checkpoint
# ===============================
WARMUP_CKPT = (
    "/path/to/workspace/"
    "ANON_USER/EfficientVit/outputs_TCGA/"
    "baseline3_brats21_source_plus_target/epoch_007.pt"
)

# =====================================================
# StageA2 config
# =====================================================
A2_CFG = {
    "margin": 0.10,
    "popsize": 12,
    "mu": 6,
    "n_gen": 15,
    "sigma0_scale": 0.2,
    "iters_eval": 500,
    "iters_refine": 1000,
    "refine_topk": 5,
}

# =====================================================
# helpers
# =====================================================
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _mk_tag(wr: float, wl: float, iters: int) -> str:
    return f"wR{wr:.3f}_wL{wl:.3f}_it{iters}"

def _run_one(
    budget_T: int,
    run_tag: str,
    w_rds: float,
    iters: int,
    seeds: str,
    warmup_ckpt: str | None = None,
    # resume_ckpt: str | None = None,
) -> Dict[str, Any]:

    w_rds = _clip01(float(w_rds))
    w_less = 1.0 - w_rds

    budget = budget_T * T
    tag = _mk_tag(w_rds, w_less, iters)

    split_dir = SPLIT_ROOT / f"{budget_T}T" / run_tag / tag
    out_dir   = OUT_ROOT   / f"{budget_T}T" / run_tag / tag
    yaml_path = out_dir / "train_config.yaml"
    train_txt = split_dir / "train_subjects.txt"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    # TCGA scores: repeat_id=None means "read directly under SCORE_ROOT/"
    rds, less = load_norm_scores(SCORE_ROOT, repeat_id=None)
    build_subset(rds, less, w_rds, w_less, budget, train_txt)

    # ckpt mode
    init_ckpt = warmup_ckpt
    ckpt_mode = "init_from_warmup"

    assert init_ckpt is not None, "init_ckpt is None (warmup_ckpt/resume_ckpt missing)"
    assert Path(init_ckpt).exists(), f"Checkpoint not found: {init_ckpt}"

    print(f"🔥 [A2] CKPT mode = {ckpt_mode}")
    print(f"🔥 [A2] Using checkpoint = {init_ckpt}")

    build_yaml(
        TEMPLATE_YAML,
        yaml_path,
        train_txt,
        out_dir,
        max_iters=iters,
        resume_ckpt=init_ckpt,
        stage="A",
    )

    run_training(yaml_path, seeds=seeds)

    dice = compute_fast_dice(out_dir, max_subjects=20)  # Stage A2
    fit = dice

    print(
        f"[A2][fast-dice] "
        f"wR={w_rds:.3f} wL={w_less:.3f} | "
        f"iters={iters} | "
        f"dice={dice:.4f}"
    )

    ckpt = out_dir / "latest.pt"
    return {
        "w_rds": w_rds,
        "w_less": w_less,
        "iters": int(iters),
        "fitness": float(fit),
        "ckpt": str(ckpt),
        "out_dir": str(out_dir),
        # "resume_from": resume_ckpt,
    }

def derive_interval_from_stageA1(stageA1_json: Path) -> Dict[str, Any]:
    d = json.loads(stageA1_json.read_text())
    finals = d.get("final_candidates", [])
    if not finals:
        raise RuntimeError("StageA1 json has no final_candidates")

    wrs = [float(w[0]) for w in finals]
    margin = float(A2_CFG["margin"])

    lo = max(0.0, min(wrs) - margin)
    hi = min(1.0, max(wrs) + margin)

    if hi - lo < 1e-3:
        c = float(np.mean(wrs))
        lo, hi = max(0.0, c - 0.05), min(1.0, c + 0.05)

    return {"lo": lo, "hi": hi, "centers": wrs}

# =====================================================
# StageA2 main
# =====================================================
def stageA2_interval_cma(
    budget_T: int,
    run_tag: str,
    stageA1_json: Path,
    seeds: str = "0",
    seed: int = 0,
) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)

    assert TEMPLATE_YAML.exists(), f"Template not found: {TEMPLATE_YAML}"
    assert SCORE_ROOT.exists(), f"Score root not found: {SCORE_ROOT}"
    assert Path(WARMUP_CKPT).exists(), f"Warmup ckpt not found: {WARMUP_CKPT}"

    interval = derive_interval_from_stageA1(stageA1_json)
    lo, hi = float(interval["lo"]), float(interval["hi"])

    popsize = int(A2_CFG["popsize"])
    mu = int(A2_CFG["mu"])
    n_gen = int(A2_CFG["n_gen"])
    it_eval = int(A2_CFG["iters_eval"])

    mean = (lo + hi) / 2.0
    sigma = max(1e-6, float(A2_CFG["sigma0_scale"]) * (hi - lo))

    all_evals = []
    gens = []
    best_so_far = None

    for g in range(n_gen):
        if g == 0:
            xs = np.linspace(lo, hi, popsize)
            rng.shuffle(xs)
        else:
            xs = rng.normal(mean, sigma, size=popsize)
            xs = np.clip(xs, lo, hi)

        print(f"\n🧬 CMA-Gen {g} | mean={mean:.4f} sigma={sigma:.4f} interval=[{lo:.3f},{hi:.3f}]")

        evals = []
        for x in xs:
            r = _run_one(
                budget_T=budget_T,
                run_tag=run_tag,
                w_rds=float(x),
                iters=it_eval,
                seeds=seeds,
                warmup_ckpt=WARMUP_CKPT,
            )
            evals.append(r)
            all_evals.append(r)

        evals.sort(key=lambda r: r["fitness"], reverse=True)
        parents = evals[:mu]

        px = np.array([p["w_rds"] for p in parents], dtype=np.float64)
        w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
        w = w / w.sum()

        mean = float(np.sum(w * px))
        sigma = float(np.std(px) + 1e-6)

        print(
            f"[A2][Gen{g}] "
            f"best_fit={parents[0]['fitness']:.4f} | "
            f"mean→{mean:.3f} sigma→{sigma:.3f}"
        )

        gens.append({
            "gen": g,
            "mean": mean,
            "sigma": sigma,
            "parents": parents,
            "candidates": xs.tolist(),
        })

        if best_so_far is None or parents[0]["fitness"] > best_so_far["fitness"]:
            best_so_far = parents[0]

    # ================= refine =================
    all_evals_sorted = sorted(all_evals, key=lambda r: r["fitness"], reverse=True)
    refine_records = []

    print(f"\n🔧 Refining top{A2_CFG['refine_topk']} to iters={A2_CFG['iters_refine']}")

    for r in all_evals_sorted[: int(A2_CFG["refine_topk"]) ]:
        rr = _run_one(
            budget_T=budget_T,
            run_tag=run_tag,
            w_rds=r["w_rds"],
            iters=int(A2_CFG["iters_refine"]),
            seeds=seeds,
            # resume_ckpt=r["ckpt"],
            warmup_ckpt=WARMUP_CKPT,
        )
        refine_records.append(rr)

    refine_records.sort(key=lambda r: r["fitness"], reverse=True)
    top3 = refine_records[:3]

    return {
        "stageA1_json": str(stageA1_json),
        "interval": interval,
        "cfg": A2_CFG,
        "gens": gens,
        "refine_records": refine_records,
        "top3": top3,
        "best_so_far_stageA2": best_so_far,
    }

# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_T", type=int, required=True)
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--stageA1_json", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = stageA2_interval_cma(
        budget_T=args.budget_T,
        run_tag=args.run_tag,
        stageA1_json=Path(args.stageA1_json),
        seeds=args.seeds,
        seed=args.seed,
    )

    save_dir = OUT_ROOT / f"{args.budget_T}T" / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "stageA2_cma.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n🏁 StageA2 Top3:")
    for r in out["top3"]:
        print(r)
