#!/bin/bash
#SBATCH --job-name=EffViT_BraTS21_IVYGAP_8D_10T
#SBATCH --output=logs/IVYGAP/train_effvit_BraTS21_IVYGAP_8D_10T_%j.out
#SBATCH --error=logs/IVYGAP/train_effvit_BraTS21_IVYGAP_8D_10T_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

# ====== Environment setup ======
source ~/envs/brain310/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# ====== Path setup ======
cd /path/to/project

# ====== Run training ======
export PYTHONPATH=.
echo "🚀 Starting EfficientViT-Seg training: IVYGAP (BraTS 8D CMA-ES) → (Target Test)"
python -u scripts/train_seg.py --config configs_IVYGAP_cma/train_config_BraTS21_8D_10T.yaml

echo "✅ Training finished! Check outputs_IVYGAP_cma/Brats21_S_8D_10T_plus_T_train"
