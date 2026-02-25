#!/bin/bash
#SBATCH --job-name=Eval_C4_random
#SBATCH --output=logs/C4/eval/random_%j.out
#SBATCH --error=logs/C4/eval/random_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
ulimit -n 4096

source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "====== C4 Random (5T / 10T) ======"

for repeat in 01 02 03 04 05
do
    # ---------------- 5T ----------------
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_random_5T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_random_repeat${repeat}_5T_plus_T_train/epoch_020.pt

    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_random2_5T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_random2_repeat${repeat}_5T_plus_T_train/epoch_020.pt

    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_random3_5T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_random3_repeat${repeat}_5T_plus_T_train/epoch_020.pt

    # ---------------- 10T ----------------
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_random_10T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_random_repeat${repeat}_10T_plus_T_train/epoch_020.pt

    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_random2_10T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_random2_repeat${repeat}_10T_plus_T_train/epoch_020.pt

    python -u eval/evaluate_2d_to_3d.py \
        --config configs_C4/repeat${repeat}/train_config_BraTS21_random3_10T.yaml \
        --checkpoint outputs_C4_new/Brats21_S_random3_repeat${repeat}_10T_plus_T_train/epoch_020.pt

done

echo "====== C4 Random Done ======"
