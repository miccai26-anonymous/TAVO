#!/usr/bin/env python3
import json, shutil
from pathlib import Path
from typing import Dict, Any

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
SPLIT_ROOT = PROJECT_ROOT / "data/splits_TCGA_stageB"
OUT_ROOT   = PROJECT_ROOT / "outputs_TCGA_stageB"

T = 50

# ===============================
# Warmup checkpoint (for corners init)
# ===============================
WARMUP_CKPT = (
    "/path/to/workspace/"
    "ANON_USER/EfficientVit/outputs_TCGA/"
    "baseline3_brats21_source_plus_target/epoch_007.pt"
)

# =====================================================
# StageB config
# =====================================================
B_CFG = {
    "iters": [8000, 16000],
    "n_keep": [3, 1],
    "max_subjects_eval": 20,
}

# =====================================================
# helpers
# =====================================================
def _mk_tag(wr, wl, iters):
    return f"wR{wr:.3f}_wL{wl:.3f}_it{iters}"

def _run_one(
    budget_T: int,
    run_tag: str,
    w_rds: float,
    iters: int,
    seeds: str,
    init_ckpt: str | None = None,     # ⭐ 第一轮初始化用
    # resume_ckpt: str | None = None,   # ⭐ 后续轮继续训练用
    max_subjects_eval: int = 20,
) -> Dict[str, Any]:

    w_less = 1.0 - float(w_rds)
    budget = int(budget_T) * T
    tag = _mk_tag(float(w_rds), float(w_less), int(iters))

    split_dir = SPLIT_ROOT / f"{budget_T}T" / run_tag / tag
    out_dir   = OUT_ROOT   / f"{budget_T}T" / run_tag / tag
    yaml_path = out_dir / "train_config.yaml"
    train_txt = split_dir / "train_subjects.txt"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    # TCGA scores: repeat_id=None means directly under SCORE_ROOT
    rds, less = load_norm_scores(SCORE_ROOT, repeat_id=None)
    build_subset(rds, less, float(w_rds), float(w_less), budget, train_txt)

    # ✅ StageB 必须保证不是“空 ckpt”启动
    
    ckpt_to_use = init_ckpt
    ckpt_mode = "init_from_fixed_ckpt"

    assert ckpt_to_use is not None, (
        "StageB got no checkpoint to start from. "
        "You must provide init_ckpt (epoch7))."
    )
    assert Path(ckpt_to_use).exists(), f"Checkpoint not found: {ckpt_to_use}"

    print(f"🔥 [B] CKPT mode = {ckpt_mode}")
    print(f"🔥 [B] Using checkpoint = {ckpt_to_use}")

    build_yaml(
        TEMPLATE_YAML,
        yaml_path,
        train_txt,
        out_dir,
        max_iters=int(iters),
        resume_ckpt=str(ckpt_to_use),
        stage="B",
    )

    run_training(yaml_path, seeds=seeds)

    dice = compute_fast_dice(out_dir, max_subjects=int(max_subjects_eval))
    ckpt = out_dir / "latest.pt"

    return {
        "w_rds": float(w_rds),
        "w_less": float(w_less),
        "iters": int(iters),
        "fitness": float(dice),
        "ckpt": str(ckpt),
        "out_dir": str(out_dir),
        # "resume_from": resume_ckpt,
        "init_from": init_ckpt,
        "max_subjects_eval": int(max_subjects_eval),
    }

# =====================================================
# StageB main
# =====================================================
def stageB_earlydice_hb(
    budget_T: int,
    run_tag: str,
    stageA2_json: Path,
    seeds: str = "0",
    max_subjects_eval: int | None = None,
) -> Dict[str, Any]:

    assert TEMPLATE_YAML.exists(), f"Template not found: {TEMPLATE_YAML}"
    assert stageA2_json.exists(), f"stageA2_json not found: {stageA2_json}"
    assert Path(WARMUP_CKPT).exists(), f"Warmup ckpt not found: {WARMUP_CKPT}"

    d = json.loads(stageA2_json.read_text())
    top3 = d["top3"]

    # ✅ pool: StageA2 top3 + corners
    # init_map: Dict[tuple[float, float], str] = {}

    pool = []
    for r in top3:
        wr = float(r["w_rds"])
        wl = float(r["w_less"])
        pool.append((wr, wl))
        # init_map[(wr, wl)] = str(r["ckpt"])   # StageA2 的 ckpt 作为 StageB round0 init

    # corners 用 warmup init
    pool += [(1.0, 0.0), (0.0, 1.0)]
    # init_map[(1.0, 0.0)] = WARMUP_CKPT
    # init_map[(0.0, 1.0)] = WARMUP_CKPT

    init_ckpt_fixed = WARMUP_CKPT

    iters_list = B_CFG["iters"]
    n_keep = B_CFG["n_keep"]

    if max_subjects_eval is None:
        max_subjects_eval = int(B_CFG.get("max_subjects_eval", 20))

    records = []
    # prev_ckpt = {}

    for ridx, (max_it, keep_k) in enumerate(zip(iters_list, n_keep)):
        print(f"\n🧪 StageB Round{ridx} | iters={max_it} | eval={len(pool)} | keep={keep_k}")

        round_res = []
        for (wr, wl) in pool:
            # 后续轮用 prev_ckpt；第一轮用 init_map
            # resume = prev_ckpt.get((wr, wl), None)
            # init_ckpt = init_map[(wr, wl)]

            res = _run_one(
                budget_T=budget_T,
                run_tag=run_tag,
                w_rds=wr,
                iters=int(max_it),
                seeds=seeds,
                init_ckpt=init_ckpt_fixed,
                # resume_ckpt=resume,
                max_subjects_eval=int(max_subjects_eval),
            )
            round_res.append(res)
            records.append(res)

        round_res.sort(key=lambda x: x["fitness"], reverse=True)
        survivors = round_res[:keep_k]

        pool = [(r["w_rds"], r["w_less"]) for r in survivors]
        # prev_ckpt = {(r["w_rds"], r["w_less"]): r["ckpt"] for r in survivors}

    best = survivors[0]

    return {
        "stageA2_json": str(stageA2_json),
        "cfg": {**B_CFG, "max_subjects_eval": int(max_subjects_eval)},
        "records": records,
        "best": best,
    }

# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_T", type=int, required=True)
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--stageA2_json", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--max_subjects_eval", type=int, default=20)
    args = ap.parse_args()

    out = stageB_earlydice_hb(
        budget_T=args.budget_T,
        run_tag=args.run_tag,
        stageA2_json=Path(args.stageA2_json),
        seeds=args.seeds,
        max_subjects_eval=args.max_subjects_eval,
    )

    save_dir = OUT_ROOT / f"{args.budget_T}T" / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "stageB_earlydice.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n🏆 StageB Best:")
    print(out["best"])
