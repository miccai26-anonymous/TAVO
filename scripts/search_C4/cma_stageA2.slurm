#!/bin/bash
#SBATCH --job-name=C4_A2_CMA_noresume_5T
#SBATCH --output=logs/C4/r5_stageA2/%x_%j.out
#SBATCH --error=logs/C4/r5_stageA2/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

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
# Run Stage A2 (Interval + CMA)
# ===============================
echo "===================================="
echo "🚀 Stage A2 | Interval CMA-ES"
echo "repeat       = 05"
echo "budget_T     = 5T"
echo "iters_eval   = 500"
echo "iters_refine = 1000"
echo "popsize      = 12"
echo "n_gen        = 15"
echo "sigma0_scale = 0.2"
echo "GPU          = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/search_C4/cma_stageA2.py \
  --repeat 5 \
  --budget_T 5 \
  --run_tag C4_A2 \
  --stageA1_json \
    outputs_C4_stageA1/repeat05/5T/C4_A1/stageA1_hyperband.json \
  --seeds 0 \
  --seed 0

echo "✅ Stage A2 finished"
