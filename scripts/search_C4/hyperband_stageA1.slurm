#!/bin/bash
#SBATCH --job-name=C4_A1_HB_noresume_10T
#SBATCH --output=logs/C4/r5_stageA1/%x_%j.out
#SBATCH --error=logs/C4/r5_stageA1/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=6:00:00

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
echo "repeat   = 05"
echo "budget_T = 10T"
echo "scheme   = Sobol + Successive Halving"
echo "iters    = [200, 500, 1200]"
echo "fitness  = loss proxy (val + gap)"
echo "GPU      = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/search_C4/hyperband_stageA1.py \
  --repeat 5 \
  --budget_T 10 \
  --run_tag C4_A1 \
  --seeds 0 \
  --seed 0

echo "✅ Stage A1 finished"
