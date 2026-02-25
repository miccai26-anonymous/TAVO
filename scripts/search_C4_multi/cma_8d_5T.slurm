#!/bin/bash
#SBATCH --job-name=C4_CMA_8D_5T
#SBATCH --output=logs/C4/stageA2_8D/repeat05/%x_%j.out
#SBATCH --error=logs/C4/stageA2_8D/repeat05/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
# Run Stage A2 (CMA)
# ===============================
echo "===================================="
echo "🚀 Stage A2 | CMA-ES"
echo "repeat       = 05"
echo "budget_T     = 5T"
echo "iters_eval   = 500"
echo "iters_refine = 1500"
echo "popsize      = 12"
echo "n_gen        = 15"
echo "sigma0 = 0.30"
echo "GPU          = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/search_C4_multi/cma_C4_8d_5T.py \
  --repeat 5 \
  --budget_T 5 \
  --run_tag C4_A2_8D \
  --seeds_eval 0 \
  --seeds_refine 0,1,2

echo "✅ Stage A2 finished"
