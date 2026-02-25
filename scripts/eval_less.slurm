#!/bin/bash
#SBATCH --job-name=Eval_C4_repeat05_3Exp
#SBATCH --output=logs/eval_C4_repeat05_3exp_%j.out
#SBATCH --error=logs/eval_C4_repeat05_3exp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
ulimit -n 4096

# ===============================
# Environment
# ===============================
source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "====== Eval repeat05 (3 experiments) ======"

# ============================================================
# 1️⃣ LESS + kmeans | repeat05 | 10T
# ============================================================
echo ">>> LESS kmeans | repeat05 | 10T"
python -u eval/evaluate_2d_to_3d.py \
  --config configs_C4/repeat05/train_config_BraTS21_less_kmeans_10T.yaml \
  --checkpoint outputs_C4/Brats21_less_kmeans_repeat05_10T_plus_T_train/best.pt


echo "====== DONE: repeat05 eval finished ======"
