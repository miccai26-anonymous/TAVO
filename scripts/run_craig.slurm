#!/bin/bash
#SBATCH --job-name=Craig_UPENN
#SBATCH --output=logs/UPENN/methods/Craig_UPENN_%j.out
#SBATCH --error=logs/UPENN/methods/Craig_UPENN_%j.err
#SBATCH --partition=largemem
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00

source ~/envs/brain310/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

cd /path/to/project

echo "🚀 Computing RDS+ scores..."

python scripts/run_craig.py \
  --target UPENN \
  --T 50 \
  --normalize

python scripts/run_craig.py \
  --target IVYGAP \
  --T 50 \
  --normalize

python scripts/run_craig.py \
  --target C5 \
  --T 50 \
  --normalize

python scripts/run_craig.py \
  --target TCGA_LGG \
  --T 50 \
  --normalize

python scripts/run_craig.py \
  --target TCGA_GBM \
  --T 50 \
  --normalize