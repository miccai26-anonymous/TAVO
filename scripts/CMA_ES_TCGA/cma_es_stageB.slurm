#!/bin/bash
#SBATCH --job-name=CMA_B_10T_iter400_v7_new
#SBATCH --output=logs/CMA_ES/TCGA/stageB/%x_%j.out
#SBATCH --error=logs/CMA_ES/TCGA/stageB/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00

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
# Run Stage B (long rerank)
# ===============================
echo "===================================="
echo "🚀 Stage B | Re-ranking Top-10"
echo "repeat   = 01"
echo "budget_T = 10T"
echo "iters    = 400"
echo "metric   = best val loss"
echo "GPU      = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/CMA_ES_TCGA/run_stageB.py

echo "✅ Stage B finished"
