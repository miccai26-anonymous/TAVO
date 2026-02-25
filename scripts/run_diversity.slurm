#!/bin/bash
#SBATCH --job-name=Diversity_TCGA
#SBATCH --output=logs/diversity_TCGA_%j.out
#SBATCH --error=logs/diversity_TCGA_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

echo "===== Activating environment ====="
source ~/envs/brain310/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

cd /path/to/project
export PYTHONPATH=.

echo "======================================="
echo "🚀 Starting diversity score computation..."
echo "======================================="

python scripts/run_diversity.py \
    --target UPENN \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0



python scripts/run_diversity.py \
    --target IVYGAP \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0

echo "======================================="
echo "🎉 diversity score computation complete!"

python scripts/run_diversity.py \
    --target C5 \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0

python scripts/run_diversity.py \
    --target TCGA_LGG \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0

python scripts/run_diversity.py \
    --target TCGA_GBM \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0