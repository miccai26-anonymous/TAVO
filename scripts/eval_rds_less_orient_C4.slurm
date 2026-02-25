#!/bin/bash
#SBATCH --job-name=Eval_C4_methods
#SBATCH --output=logs/C4/eval/methods_%j.out
#SBATCH --error=logs/C4/eval/methods_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
ulimit -n 4096

source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "====== C4 RDS / LESS / ORIENT ======"

for repeat in 01 02 03 04 05
do
    # ===============================
    # 5T
    # ===============================

    # RDS
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_rdsplus_5T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_rds_repeat${repeat}_5T_plus_T_train/epoch_020.pt

    # LESS
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_less_5T.yaml \
        --checkpoint outputs_C4_new/Brats21_less_repeat${repeat}_5T_plus_T_train/epoch_020.pt

    # ORIENT
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_orient_5T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_orient_repeat${repeat}_epoch3_5T_plus_T_train/epoch_020.pt


    # ===============================
    # 10T
    # ===============================

    # RDS
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_rdsplus_10T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_rds_repeat${repeat}_10T_plus_T_train/epoch_020.pt

    # LESS
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_less_10T.yaml \
        --checkpoint outputs_C4_new/Brats21_less_repeat${repeat}_10T_plus_T_train/epoch_020.pt

    # ORIENT
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_orient_10T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_orient_repeat${repeat}_epoch3_10T_plus_T_train/epoch_020.pt

done

echo "====== C4 Methods Done ======"
