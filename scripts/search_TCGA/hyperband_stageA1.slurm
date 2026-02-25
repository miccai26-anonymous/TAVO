#!/bin/bash
#SBATCH --job-name=TCGA_A1_HB_noresume_10T
#SBATCH --output=logs/TCGA/stageA1/%x_%j.out
#SBATCH --error=logs/TCGA/stageA1/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00

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
# Run StageA1 (Hyperband)
# ===============================
echo "===================================="
echo "🚀 Stage A1 | Hyperband (RDS + LESS)"
echo "budget_T = 10T"
echo "scheme   = Sobol + Successive Halving"
echo "iters    = [200, 500, 1000]"
echo "fitness  = loss proxy (val + gap)"
echo "GPU      = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/search_TCGA/hyperband_stageA1.py \
  --budget_T 10 \
  --run_tag TCGA_A1 \
  --seeds "0" \
  --seed 0


echo "✅ Stage A1 finished"
