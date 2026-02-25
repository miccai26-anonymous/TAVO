#!/bin/bash
#SBATCH --job-name=Orient_C5
#SBATCH --output=logs/C5/methods/orient_C5_%j.out
#SBATCH --error=logs/C5/methods/orient_C5_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

echo "===== Activating environment ====="
source ~/envs/brain310/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

cd /path/to/project
export PYTHONPATH=.

echo "=========================================="
echo "🚀 Starting ORIENT subsets selection..."
echo "=========================================="

# echo "=========================================="
# echo "🚀 Starting UPENN"
# echo "=========================================="

# python scripts/run_orient.py \
#   --target UPENN \
#   --T 50 \
#   --eta 1.0 \
#   --warmup_ckpt outputs_UPENN/baseline3_brats21_source_plus_target/epoch_006.pt

# echo "=========================================="
# echo "🎉 ORIENT subsets selection complete!"
# echo "Saved to: data/splits_UPENN_orient/"

# echo "=========================================="
# echo "🚀 Starting IVYGAP"
# echo "=========================================="

# python scripts/run_orient.py \
#   --target IVYGAP \
#   --T 50 \
#   --eta 1.0 \
#   --warmup_ckpt outputs_IVYGAP/baseline3_brats21_source_plus_target/epoch_006.pt

# echo "=========================================="
# echo "🎉 ORIENT subsets selection complete!"
# echo "Saved to: data/splits_IVYGAP_orient/"

echo "=========================================="
echo "🚀 Starting C5"
echo "=========================================="

python scripts/run_orient.py \
  --target C5 \
  --T 50 \
  --eta 1.0 \
  --warmup_ckpt outputs_C5/baseline3_brats21_source_plus_target/epoch_011.pt

echo "=========================================="
echo "🎉 ORIENT subsets selection complete!"
echo "Saved to: data/splits_C5_orient/"


# echo "=========================================="
# echo "🚀 Starting TCGA_LGG"
# echo "=========================================="

# python scripts/run_orient.py \
#   --target TCGA_LGG \
#   --T 50 \
#   --eta 1.0 \
#   --warmup_ckpt outputs_TCGA_LGG/baseline3_brats21_source_plus_target/epoch_013.pt

# echo "=========================================="
# echo "🎉 ORIENT subsets selection complete!"
# echo "Saved to: data/splits_TCGA_LGG_orient/"


# echo "=========================================="
# echo "🚀 Starting TCGA_GBM"
# echo "=========================================="

# python scripts/run_orient.py \
#   --target TCGA_GBM \
#   --T 50 \
#   --eta 1.0 \
#   --warmup_ckpt outputs_TCGA_GBM/baseline3_brats21_source_plus_target/epoch_014.pt

# echo "=========================================="
# echo "🎉 ORIENT subsets selection complete!"
# echo "Saved to: data/splits_TCGA_GBM_orient/"

