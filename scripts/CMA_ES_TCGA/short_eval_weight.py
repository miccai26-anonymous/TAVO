#!/usr/bin/env python3
import os
import yaml
import json
import subprocess
import numpy as np
from pathlib import Path
import shutil


# =====================================================
# Global Config
# =====================================================
PROJECT_ROOT = Path("/path/to/project")
TRAIN_ENTRY = ["python", "scripts/train_seg_short.py"]

TEMPLATE_YAML = PROJECT_ROOT / "configs_TCGA_cma/template.yaml"

SCORE_ROOT = PROJECT_ROOT / "data/splits_TCGA_mix_scores"
CMA_SPLIT_ROOT = PROJECT_ROOT / "data/splits_TCGA_cma"
CMA_OUTPUT_ROOT = PROJECT_ROOT / "outputs_TCGA_cma"

MAX_ITERS = 400   # short-run
T = 50            # target size

DEVICE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "NUMEXPR_MAX_THREADS": "8",
}


# =====================================================
# Utils
# =====================================================
def load_norm_scores():
    rds = np.load(SCORE_ROOT / "rds_norm_dict.npy", allow_pickle=True).item()
    less = np.load(SCORE_ROOT / "less_norm_dict.npy", allow_pickle=True).item()
    return rds, less


def build_subset(rds, less, w_rds, w_less, budget, out_txt):
    scores = {k: w_rds * rds[k] + w_less * less[k] for k in rds}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [k for k, _ in ranked[:budget]]

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write("\n".join(selected))

    return selected


def build_yaml(template_yaml, out_yaml, train_subjects_txt, output_dir):
    with open(template_yaml) as f:
        cfg = yaml.safe_load(f)

    # --- CMA overrides ---
    cfg["data"]["domains"][0]["split_txt"] = str(train_subjects_txt.parent)
    cfg["trainer"]["max_iters"] = MAX_ITERS
    cfg["training"]["save_dir"] = str(output_dir)

    # 🔑 control skip_empty explicitly (Stage-A)
    cfg["data"]["skip_empty_train"] = True
    cfg["data"]["skip_empty_val"] = True

    # --- logging ---
    cfg.setdefault("logging", {})
    cfg["logging"]["save_loss_curve"] = True

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f)


def run_training(yaml_path, seeds="0"):
    env = os.environ.copy()
    env.update(DEVICE_ENV)

    # 🔥 ensure models/ is discoverable
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = TRAIN_ENTRY + ["--config", str(yaml_path), "--seeds", str(seeds)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def load_losses(output_dir):
    # NEW: prefer multi-seed mean
    mean_train = output_dir / "train_losses_mean.npy"
    mean_val = output_dir / "val_losses_mean.npy"
    if mean_train.exists() and mean_val.exists():
        train = np.load(mean_train)
        val = np.load(mean_val)
        return train, val

    # old formats
    if (output_dir / "train_losses.npy").exists():
        train = np.load(output_dir / "train_losses.npy")
        val = np.load(output_dir / "val_losses.npy")
        return train, val

    if (output_dir / "losses.json").exists():
        with open(output_dir / "losses.json") as f:
            d = json.load(f)
        return np.array(d["train"]), np.array(d["val"])

    raise RuntimeError(f"No loss files found in {output_dir}")


def slope(x):
    t = np.arange(len(x))
    return np.polyfit(t, x, 1)[0]


# =========================Version 4.1=========================
def compute_fitness(train_losses, val_losses, alpha=0.2, top_ratio=0.1):
    """
    Stage-A proxy fitness (v4.1):
    - primary: mean of top-k best val losses (robust to noise)
    - secondary: mild generalization gap (aligned with v4)
    larger = better
    """
    val_losses = np.asarray(val_losses)
    train_losses = np.asarray(train_losses)

    # ---------- top-k val ----------
    n = len(val_losses)
    k = max(1, int(np.ceil(n * top_ratio)))

    # min k val loss
    best_val_losses = np.partition(val_losses, k - 1)[:k]
    val_top_mean = float(np.mean(best_val_losses))

    # ---------- train reference ----------
    train_best = float(np.min(train_losses))

    # ---------- generalization gap ----------
    gap = max(0.0, val_top_mean - train_best)

    fitness = -val_top_mean - alpha * gap

    # ---------- debug ----------
    print(
        f"[DEBUG v4.1] "
        f"val_top_mean={val_top_mean:.4f} "
        f"(k={k}/{n}) "
        f"train_min={train_best:.4f} "
        f"gap={gap:.4f} "
        f"fitness={fitness:.4f}"
    )
    return fitness


# =====================================================
# Internal runner (shared by Stage-A / Stage-B)
# =====================================================
def _run_once(budget_T, w_rds, w_less, run_tag, seeds="0"):
    """
    Build subset + yaml + run training once, then load losses.
    Returns: (train_losses, val_losses, out_dir)
    """
    assert abs(w_rds + w_less - 1.0) < 1e-6

    budget = budget_T * T
    tag = f"wR{w_rds:.3f}_wL{w_less:.3f}"

    split_dir = CMA_SPLIT_ROOT / f"{budget_T}T" / run_tag / tag
    out_dir = CMA_OUTPUT_ROOT / f"{budget_T}T" / run_tag / tag

    yaml_path = split_dir / "train_config.yaml"
    train_txt = split_dir / "train_subjects.txt"

    # clean old outputs
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # build subset
    rds, less = load_norm_scores()
    build_subset(rds, less, w_rds, w_less, budget, train_txt)

    # build yaml
    build_yaml(TEMPLATE_YAML, yaml_path, train_txt, out_dir)

    # train
    run_training(yaml_path, seeds=seeds)

    # losses
    train_losses, val_losses = load_losses(out_dir)
    return train_losses, val_losses, out_dir


# =====================================================
# Main CMA evaluation (Stage-A)
# =====================================================
def evaluate_weight(budget_T, w_rds, w_less, run_tag, seeds="0"):
    train_losses, val_losses, _ = _run_once(
        budget_T=budget_T,
        w_rds=w_rds,
        w_less=w_less,
        run_tag=run_tag,
        seeds=seeds,
    )
    return compute_fitness(train_losses, val_losses)


# =====================================================
# Stable CMA evaluation (multi-seed in ONE run)
# =====================================================
def evaluate_weight_stable(budget_T, w_rds, w_less, run_tag, seeds="0"):
    # multi-seed averaging happens inside train_seg_short.py
    return float(evaluate_weight(budget_T, w_rds, w_less, run_tag, seeds=seeds))


# =====================================================
# Stage-B evaluation (longer, no repeat)
# =====================================================
def evaluate_weight_stageB(budget_T, w_rds, w_less, run_tag):
    global MAX_ITERS
    old_iters = MAX_ITERS
    MAX_ITERS = 2000

    try:
        train_losses, val_losses, _ = _run_once(
            budget_T=budget_T,
            w_rds=w_rds,
            w_less=w_less,
            run_tag=run_tag,
        )

        # ✅ Stage B fitness = best val loss only (larger is better -> negative loss)
        print(
            f"[StageB DEBUG] val_min={float(np.min(val_losses)):.6f}  "
            f"fitness={-float(np.min(val_losses)):.6f}"
        )
        return -float(np.min(val_losses))

    finally:
        MAX_ITERS = old_iters


# =====================================================
# Quick test
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_T", type=int, required=True)
    parser.add_argument("--w_rds", type=float, required=True)
    parser.add_argument("--w_less", type=float, required=True)
    parser.add_argument("--run_tag", type=str, required=True)
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--seeds", type=str, default="0")  # NEW
    args = parser.parse_args()

    if args.stage == "A":
        f = evaluate_weight(
            budget_T=args.budget_T,
            w_rds=args.w_rds,
            w_less=args.w_less,
            run_tag=args.run_tag,
            seeds=args.seeds,
        )
    else:
        f = evaluate_weight_stageB(
            budget_T=args.budget_T,
            w_rds=args.w_rds,
            w_less=args.w_less,
            run_tag=args.run_tag,
            seeds=args.seeds,
        )

    print("Fitness:", f)
