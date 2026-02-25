#!/bin/bash
#SBATCH --job-name=CMA_short_eval
#SBATCH --output=logs/CMA_ES/%x_%j.out
#SBATCH --error=logs/CMA_ES/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# ===============================
# Environment
# ===============================
source ~/envs/brain310/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# ===============================
# Project root
# ===============================
cd /path/to/project
export PYTHONPATH=.

# ===============================
# Args from sbatch
# ===============================
REPEAT=$1        # e.g. 01
BUDGET_T=$2      # 5 or 10
W_RDS=$3         # 0.6
W_LESS=$4        # 0.4

echo "===================================="
echo "🚀 CMA-ES Short Eval (GPU)"
echo "repeat   = ${REPEAT}"
echo "budget_T = ${BUDGET_T}"
echo "w_rds    = ${W_RDS}"
echo "w_less   = ${W_LESS}"
echo "===================================="

# ===============================
# Run short eval
# ===============================
python scripts/CMA_ES/short_eval_weight.py \
  --repeat ${REPEAT} \
  --budget_T ${BUDGET_T} \
  --w_rds ${W_RDS} \
  --w_less ${W_LESS}

echo "✅ Short eval finished"
