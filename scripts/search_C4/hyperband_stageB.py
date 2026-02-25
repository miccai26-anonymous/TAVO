#!/usr/bin/env python3
import json, shutil
from pathlib import Path
from typing import Dict, Any, List

from scripts.search_C4.utils_cma import (
    load_norm_scores,
    build_subset,
    build_yaml,
    run_training,
    compute_fast_dice,
)

# =====================================================
# Paths
# =====================================================
PROJECT_ROOT = Path("/path/to/project")
TEMPLATE_YAML = PROJECT_ROOT / "configs_C4_cma/template.yaml"

SCORE_ROOT = PROJECT_ROOT / "data/splits_C4_mix_scores"
SPLIT_ROOT = PROJECT_ROOT / "data/splits_C4_stageB"
OUT_ROOT   = PROJECT_ROOT / "outputs_C4_stageB"

T = 50

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
# StageB config
# =====================================================
B_CFG = {
    "iters": [8000, 16000],
    "n_keep": [3, 1],
}

# =====================================================
# helpers
# =====================================================
def _mk_tag(wr, wl, iters):
    return f"wR{wr:.3f}_wL{wl:.3f}_it{iters}"

def _run_one(
    repeat_id: int,
    budget_T: int,
    run_tag: str,
    w_rds: float,
    iters: int,
    seeds: str,
    init_ckpt: str | None = None,
    # resume_ckpt: str | None = None,
) -> Dict[str, Any]:
    w_less = 1.0 - w_rds
    budget = budget_T * T
    tag = _mk_tag(w_rds, w_less, iters)

    split_dir = SPLIT_ROOT / f"repeat{repeat_id:02d}" / f"{budget_T}T" / run_tag / tag
    out_dir   = OUT_ROOT   / f"repeat{repeat_id:02d}" / f"{budget_T}T" / run_tag / tag
    yaml_path = out_dir / "train_config.yaml"
    train_txt = split_dir / "train_subjects.txt"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    rds, less = load_norm_scores(SCORE_ROOT, repeat_id)
    build_subset(rds, less, w_rds, w_less, budget, train_txt)

    
    ckpt = init_ckpt
    ckpt_mode = "init_from_warmup"

    print(f"🔥 [StageB] CKPT mode = {ckpt_mode}")
    print(f"🔥 [StageB] Using checkpoint = {ckpt}")

    build_yaml(
        TEMPLATE_YAML,
        yaml_path,
        train_txt,
        out_dir,
        max_iters=iters,
        resume_ckpt=ckpt,
        stage="B",
    )

    run_training(yaml_path, seeds=seeds)

    dice = compute_fast_dice(out_dir, max_subjects=20)   # ⭐ StageB max_subjects
    ckpt = out_dir / "latest.pt"

    return {
        "w_rds": w_rds,
        "w_less": w_less,
        "iters": iters,
        "fitness": float(dice),
        "ckpt": str(ckpt),
        "out_dir": str(out_dir),
        # "resume_from": resume_ckpt,
    }

# =====================================================
# StageB main
# =====================================================
def stageB_earlydice_hb(
    repeat_id: int,
    budget_T: int,
    run_tag: str,
    stageA2_json: Path,
    seeds: str = "0",
) -> Dict[str, Any]:

    warmup_ckpt = WARMUP_CKPT_MAP[f"repeat{repeat_id:02d}"]
    assert Path(warmup_ckpt).exists(), f"Warmup ckpt not found: {warmup_ckpt}"

    d = json.loads(stageA2_json.read_text())
    top3 = d["top3"]

    # candidates: StageA2 top3 + corners
    pool = [(r["w_rds"], r["w_less"]) for r in top3]
    pool += [(1.0, 0.0), (0.0, 1.0)]

    iters_list = B_CFG["iters"]
    n_keep = B_CFG["n_keep"]

    records = []
    # prev_ckpt = {}

    for ridx, (max_it, keep_k) in enumerate(zip(iters_list, n_keep)):
        print(f"\n🧪 StageB Round{ridx} | iters={max_it} | eval={len(pool)} | keep={keep_k}")

        round_res = []
        for (wr, wl) in pool:
            # ckpt = prev_ckpt.get((wr, wl), None)
            res = _run_one(
                repeat_id=repeat_id,
                budget_T=budget_T,
                run_tag=run_tag,
                w_rds=wr,
                iters=max_it,
                seeds=seeds,
                init_ckpt=warmup_ckpt,
                # resume_ckpt=ckpt,
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
        "records": records,
        "best": best,
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
    ap.add_argument("--stageA2_json", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0")
    args = ap.parse_args()

    out = stageB_earlydice_hb(
        repeat_id=args.repeat,
        budget_T=args.budget_T,
        run_tag=args.run_tag,
        stageA2_json=Path(args.stageA2_json),
        seeds=args.seeds,
    )

    save_dir = OUT_ROOT / f"repeat{args.repeat:02d}" / f"{args.budget_T}T" / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "stageB_earlydice.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n🏆 StageB Best:")
    print(out["best"])
