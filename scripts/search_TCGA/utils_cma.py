#!/usr/bin/env python3
import os, json, yaml, shutil, subprocess
import numpy as np
from pathlib import Path

# =====================================================
# Global
# =====================================================
PROJECT_ROOT = Path(
    "/path/to/project"
)
TRAIN_ENTRY = ["python", "scripts/train_seg_short.py"]

DEVICE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "NUMEXPR_MAX_THREADS": "8",
}

# =====================================================
# Scores
# =====================================================
def load_norm_scores(score_root: Path, repeat_id: int):
    if repeat_id is None:
        base = score_root
    else:
        base = score_root / f"repeat{repeat_id:02d}"
    rds  = np.load(base / "rds_norm_dict.npy", allow_pickle=True).item()
    less = np.load(base / "less_norm_dict.npy", allow_pickle=True).item()
    return rds, less

# =====================================================
# Build subset
# =====================================================
def build_subset(rds, less, w_rds, w_less, budget, out_txt: Path):
    scores = {k: w_rds * rds[k] + w_less * less[k] for k in rds}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [k for k, _ in ranked[:budget]]

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(selected))
    return selected

# =====================================================
# Set domain split in YAML cfg
# =====================================================
def set_domain_split(cfg, domain_name: str, split_txt: Path):
    found = False
    for dom in cfg["data"]["domains"]:
        if dom.get("name") == domain_name:
            dom["split_txt"] = str(split_txt)
            found = True
    assert found, f"Domain '{domain_name}' not found in cfg['data']['domains']"

# =====================================================
# YAML builder (⭐支持 resume checkpoint)
# =====================================================
def build_yaml(
    template_yaml: Path,
    out_yaml: Path,
    train_subjects_txt: Path,
    output_dir: Path,
    max_iters: int,
    resume_ckpt: str | None = None,
    stage: str = "A",
):
    cfg = yaml.safe_load(template_yaml.read_text())

    # ---- override data split ----
    domain_names = [d.get("name") for d in cfg["data"]["domains"]]
    assert "source" in domain_names, \
        f"Expected domain named 'source', got {domain_names}"
    assert "target" in domain_names, \
    f"Expected domain named 'target', got {domain_names}"
    
    # cfg["data"]["domains"][0]["split_txt"] = str(train_subjects_txt.parent)
    set_domain_split(cfg, "source", train_subjects_txt.parent)
    
    # ---- iteration budget ----
    cfg["trainer"]["max_iters"] = int(max_iters)

    # ---- output ----
    cfg["training"]["save_dir"] = str(output_dir)

    # ---- Stage-A settings ----
    if stage == "A":
        cfg["data"]["skip_empty_train"] = True
        cfg["data"]["skip_empty_val"] = True
    if stage == "B":
        cfg["data"]["skip_empty_train"] = True
        cfg["data"]["skip_empty_val"] = False

    # ---- resume checkpoint (关键) ----
    if resume_ckpt is not None:
        cfg["warmup"] = {"checkpoint": str(resume_ckpt)}
    else:
        cfg.pop("warmup", None)

    # ---- logging ----
    cfg.setdefault("logging", {})
    cfg["logging"]["save_loss_curve"] = True

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg))

# =====================================================
# Run training
# =====================================================
def run_training(yaml_path: Path, seeds="0"):
    env = os.environ.copy()
    env.update(DEVICE_ENV)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = TRAIN_ENTRY + ["--config", str(yaml_path), "--seeds", str(seeds)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

# # =====================================================
# # Load losses
# # =====================================================
# def load_losses(output_dir: Path):
#     mean_train = output_dir / "train_losses_mean.npy"
#     mean_val   = output_dir / "val_losses_mean.npy"

#     if mean_train.exists() and mean_val.exists():
#         return np.load(mean_train), np.load(mean_val)

#     if (output_dir / "train_losses.npy").exists():
#         return (
#             np.load(output_dir / "train_losses.npy"),
#             np.load(output_dir / "val_losses.npy"),
#         )

#     if (output_dir / "losses.json").exists():
#         d = json.loads((output_dir / "losses.json").read_text())
#         return np.array(d["train"]), np.array(d["val"])

#     raise RuntimeError(f"No loss files found in {output_dir}")

# # =====================================================
# # Fitness (low-K Mean + gap)
# # =====================================================
# def compute_fitness(train_losses, val_losses, alpha=0.2, top_ratio=0.1):
#     val_losses   = np.asarray(val_losses)
#     train_losses = np.asarray(train_losses)

#     n = len(val_losses)
#     k = max(1, int(np.ceil(n * top_ratio)))

#     best_val_losses = np.partition(val_losses, k - 1)[:k]
#     val_top_mean = float(np.mean(best_val_losses))

#     train_best = float(np.min(train_losses))
#     gap = max(0.0, val_top_mean - train_best)

#     fitness = -val_top_mean - alpha * gap

#     return fitness, {
#         "train_best": train_best,
#         "val_top_mean": val_top_mean,
#         "gap": gap,
#         "k": k,
#     }

# =====================================================
# Fitness (time-tail eval mean + gap)
# =====================================================
# def compute_fitness(train_losses,
#                     val_eval_means,
#                     alpha=0.5,
#                     tail_ratio=0.2):
#     """
#     train_losses     : np.ndarray, shape [num_train_iters]
#     val_eval_means   : np.ndarray, shape [num_evals]
#                        each entry = mean val loss of one eval
#     """

#     train_losses     = np.asarray(train_losses)
#     val_eval_means   = np.asarray(val_eval_means)

#     n = len(val_eval_means)
#     k = max(1, int(np.ceil(n * tail_ratio)))

#     # ---- time tail ----
#     val_tail = val_eval_means[-k:]
#     val_tail_mean = float(np.mean(val_tail))

#     # ---- train reference ----
#     train_best = float(np.min(train_losses))

#     gap = max(0.0, val_tail_mean - train_best)

#     fitness = -val_tail_mean - alpha * gap

#     return fitness, {
#         "train_best": train_best,
#         "val_tail_mean": val_tail_mean,
#         "gap": gap,
#         "k": k,
#         "n_eval": n,
#     }

# =====================================================
# Fitness: Val Mean + Gap Penalty (Stage A clean proxy)
# =====================================================
# def compute_fitness(train_losses, val_losses, alpha=0.2):
#     train_losses = np.asarray(train_losses)
#     val_losses   = np.asarray(val_losses)

#     val_mean   = float(np.mean(val_losses))
#     train_best = float(np.min(train_losses))

#     gap = max(0.0, val_mean - train_best)

#     fitness = -val_mean - alpha * gap

#     return fitness, {
#         "val_mean": val_mean,
#         "train_best": train_best,
#         "gap": gap,
#     }


# =====================================================
# Fitness Early Dice
# =====================================================
def compute_fast_dice(
    out_dir: Path,
    max_subjects: int,
):
    """
    Unified fitness for Stage A1 / A2 / B.
    Returns dice_macro (higher is better).
    """

    out_dir = Path(out_dir)

    cfg_path = out_dir / "train_config.yaml"
    ckpt_path = out_dir / "latest.pt"
    dice_json = out_dir / f"fast_dice_{max_subjects}.json"

    if not cfg_path.exists():
        raise RuntimeError(f"Missing train_config.yaml in {out_dir}")
    if not ckpt_path.exists():
        raise RuntimeError(f"Missing latest.pt in {out_dir}")

    # 如果已经算过 early dice，就直接读（避免重复 eval）
    if dice_json.exists():
        d = json.loads(dice_json.read_text())
        return float(d["dice_macro"])

    # 调用 StageB eval 脚本
    eval_script = PROJECT_ROOT / "scripts" / "search_TCGA" / "eval_early_dice_stageB.py"

    cmd = [
        "python",
        str(eval_script),
        "--config", str(cfg_path),
        "--checkpoint", str(ckpt_path),
        "--out_dir", str(out_dir),
        "--max_subjects", str(max_subjects),
        "--out_json", str(dice_json),
    ]

    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=os.environ,
        check=True,
    )

    if not dice_json.exists():
        raise RuntimeError(f"{dice_json.name} not found after eval in {out_dir}")

    d = json.loads(dice_json.read_text())
    return float(d["dice_macro"])
