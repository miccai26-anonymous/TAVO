#!/bin/bash
#SBATCH --job-name=Eval_TCGA_methods
#SBATCH --output=logs/TCGA/eval/methods_%j.out
#SBATCH --error=logs/TCGA/eval/methods_%j.err
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

echo "====== TCGA Methods ======"

for budget in 5T 10T
do
    python -u eval/evaluate_2d_to_3d.py \
        --config configs_TCGA/train_config_BraTS21_rdsplus_${budget}.yaml \
        --checkpoint outputs_TCGA/Brats21_S_rdsplus_${budget}_plus_T_train/epoch_020.pt

    python -u eval/evaluate_2d_to_3d.py \
        --config configs_TCGA/train_config_BraTS21_less_${budget}.yaml \
        --checkpoint outputs_TCGA/Brats21_S_less_${budget}_plus_T_train/epoch_020.pt

    python -u eval/evaluate_2d_to_3d.py \
        --config configs_TCGA/train_config_BraTS21_orient_${budget}.yaml \
        --checkpoint outputs_TCGA/Brats21_S_orient_${budget}_plus_T_train/epoch_020.pt
done

echo "====== TCGA Methods Done ======"
