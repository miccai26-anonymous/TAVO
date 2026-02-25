#!/bin/bash
#SBATCH --job-name=TCGA_LGG_CMA_3D_10T
#SBATCH --output=logs/TCGA_LGG/stageA2_3D/%x_%j.out
#SBATCH --error=logs/TCGA_LGG/stageA2_3D/%x_%j.err
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
echo "🚀 Stage A2 | Interval CMA-ES"
echo "budget_T     = 10T"
echo "iters_eval   = 1000"
echo "iters_refine = 2000"
echo "popsize      = 12"
echo "n_gen        = 12"
echo "sigma0_scale = 0.3"
echo "GPU          = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/search_multi/cma_TCGA_LGG_3d_10T.py \
  --budget_T 10 \
  --run_tag TCGA_LGG_A2_3D \
  --seeds_eval 0 \
  --seeds_refine 0,1,2

echo "✅ Stage A2 finished"
