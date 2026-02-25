#!/bin/bash
#SBATCH --job-name=C4_B_EarlyDice_noresume_5T
#SBATCH --output=logs/C4/stageB/%x_%j.out
#SBATCH --error=logs/C4/stageB/%x_%j.err
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
# Run Stage B (Early Dice HB)
# ===============================
echo "===================================="
echo "🚀 Stage B | Early Dice Hyperband"
echo "repeat       = 01"
echo "budget_T     = 5T"
echo "round0 iters = 8000   (~2 epoch)"
echo "round1 iters = 16000   (~4 epochs total)"
echo "keep         = 5 → 3 → 1"
echo "fitness      = fast dice (early 3D)"
echo "GPU          = $CUDA_VISIBLE_DEVICES"
echo "===================================="

python scripts/search_C4/hyperband_stageB.py \
  --repeat 1 \
  --budget_T 5 \
  --run_tag C4_B \
  --stageA2_json \
    outputs_C4_stageA2/repeat01/5T/C4_A2/stageA2_cma.json \
  --seeds 0

echo "✅ Stage B finished"
