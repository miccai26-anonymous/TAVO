#!/bin/bash
#SBATCH --job-name=Kcenter_TCGA
#SBATCH --output=logs/kcenter_TCGA_%j.out
#SBATCH --error=logs/kcenter_TCGA_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
echo "🚀 Starting kcenter score computation..."
echo "======================================="

python scripts/run_kcenter.py \
    --target UPENN \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0


python scripts/run_kcenter.py \
    --target IVYGAP \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0



echo "======================================="
echo "🎉 kcenter score computation complete!"

python scripts/run_kcenter.py \
    --target C5 \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0

python scripts/run_kcenter.py \
    --target TCGA_LGG \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0

python scripts/run_kcenter.py \
    --target TCGA_GBM \
    --T 50 \
    --max_rank 750 \
    --normalize \
    --seed 0