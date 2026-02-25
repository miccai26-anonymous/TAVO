#!/bin/bash
#SBATCH --job-name=Gradmatch_UPENN
#SBATCH --output=logs/UPENN/methods/gradmatch_UPENN_%j.out
#SBATCH --error=logs/UPENN/methods/gradmatch_UPENN_%j.err
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
echo "🚀 Starting gradmatch score computation..."
echo "======================================="

python scripts/run_gradmatch.py \
  --target UPENN \
  --T 50 \
  --normalize

python scripts/run_gradmatch.py \
  --target IVYGAP \
  --T 50 \
  --normalize


echo "======================================="
echo "🎉 gradmatch score computation complete!"
echo "Results saved to: results/gradmatch_subsets/"

python scripts/run_gradmatch.py \
  --target C5 \
  --T 50 \
  --normalize

python scripts/run_gradmatch.py \
  --target TCGA_LGG \
  --T 50 \
  --normalize

python scripts/run_gradmatch.py \
  --target TCGA_GBM \
  --T 50 \
  --normalize