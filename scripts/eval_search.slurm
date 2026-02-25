#!/bin/bash
#SBATCH --job-name=Eval_TCGA_craig
#SBATCH --output=logs/eval_TCGA_craig_%j.out
#SBATCH --error=logs/eval_TCGA_craig_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
ulimit -n 4096

# ===============================
# Environment
# ===============================
source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "===================================="
echo "🚀  Evaluating C4 + TCGA (5T & 10T)"
echo "===================================="


# =====================================================
# ======================= C4 5T =======================
# =====================================================
echo "========== C4 5T =========="

python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS21_craig_10T.yaml \
    --checkpoint outputs_TCGA/Brats21_S_craig_10T_plus_T_train/best.pt

# HB-2
python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS21_craig_10T.yaml \
    --checkpoint outputs_TCGA/Brats21_S_craig_10T_plus_T_train/epoch_016.pt

# HB-1
python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS21_craig_10T.yaml \
    --checkpoint outputs_TCGA/Brats21_S_craig_10T_plus_T_train/epoch_018.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS21_craig_10T.yaml \
    --checkpoint outputs_TCGA/Brats21_S_craig_10T_plus_T_train/epoch_019.pt



# =====================================================
# ======================= C4 10T ======================
# =====================================================
echo "========== C4 10T =========="

# HB-1
# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_TCGA/repeat01/train_config_BraTS21_rds_less_hb_1_10T.yaml \
#     --checkpoint outputs_C4_HB/Brats21_rds_less_HB_1_repeat01_10T_plus_T_train/epoch_020.pt

# # HB-2
# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_TCGA/repeat01/train_config_BraTS21_craig_10T.yaml \
#     --checkpoint outputs_C4_HB/Brats21_craig_repeat01_10T_plus_T_train/epoch_020.pt

# # HB-main
# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_TCGA/repeat01/train_config_BraTS21_rds_less_hb_10T.yaml \
#     --checkpoint outputs_C4_HB/Brats21_rds_less_HB_repeat01_10T_plus_T_train/epoch_020.pt


# # =====================================================
# # ======================= TCGA 5T =====================
# # =====================================================
# echo "========== TCGA 5T =========="

# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_TCGA_hb/train_config_BraTS21_rds_less_HB_5T.yaml \
#     --checkpoint outputs_TCGA_HB/Brats21_S_less_5T_plus_T_train/epoch_020.pt


# # =====================================================
# # ======================= TCGA 10T ====================
# # =====================================================
# echo "========== TCGA 10T =========="

# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_TCGA_hb/train_config_BraTS21_rds_less_HB_10T.yaml \
#     --checkpoint outputs_TCGA/Brats21_S_less_10T_plus_T_train/epoch_020.pt

echo "===================================="
echo "✅ ALL HB Evaluations Finished"
echo "===================================="
