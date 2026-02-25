#!/bin/bash
#SBATCH --job-name=Build_Norm_Score_Dicts_TCGA
#SBATCH --output=logs/build_norm_score_dicts_TCGA_%j.out
#SBATCH --error=logs/build_norm_score_dicts_TCGA_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00

ulimit -n 4096

# ============================================================
# Environment
# ============================================================
source ~/envs/brain310/bin/activate
export PYTHONUNBUFFERED=1

cd /path/to/project
export PYTHONPATH=.

python scripts/build_all_norm_score_dicts.py \
    --target TCGA

echo "=========================================="
echo "🎉 DONE"
echo "=========================================="

python scripts/build_all_norm_score_dicts.py \
    --target C5

python scripts/build_all_norm_score_dicts.py \
    --target TCGA_LGG

python scripts/build_all_norm_score_dicts.py \
    --target TCGA_GBM