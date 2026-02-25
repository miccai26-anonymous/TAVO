#!/bin/bash
#SBATCH --job-name=Eval_Baseline
#SBATCH --output=logs/C4/eval/baseline/eval_baseline_%j.out
#SBATCH --error=logs/C4/eval/baseline/eval_baseline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
ulimit -n 4096

# ===============================
# Environment
# ===============================
source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "====== Running C4 Baseline Evaluations ======"

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat01/train_config_BraTS2021_Target_train.yaml \
    --checkpoint outputs_C4_new/target_only_repeat01/epoch_060.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat02/train_config_BraTS2021_Target_train.yaml \
    --checkpoint outputs_C4_new/target_only_repeat02/epoch_060.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat03/train_config_BraTS2021_Target_train.yaml \
    --checkpoint outputs_C4_new/target_only_repeat03/epoch_060.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat04/train_config_BraTS2021_Target_train.yaml \
    --checkpoint outputs_C4_new/target_only_repeat04/epoch_060.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat05/train_config_BraTS2021_Target_train.yaml \
    --checkpoint outputs_C4_new/target_only_repeat05/epoch_060.pt

# ===============================

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat01/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_C4_new/source_only_01/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat02/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_C4_new/source_only_02/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat03/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_C4_new/source_only_03/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat04/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_C4_new/source_only_04/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat05/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_C4_new/source_only_05/epoch_020.pt

# ===============================

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat01/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_C4_new/source_plus_target_repeat01/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat02/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_C4_new/source_plus_target_repeat02/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat03/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_C4_new/source_plus_target_repeat03/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat04/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_C4_new/source_plus_target_repeat04/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4/repeat05/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_C4_new/source_plus_target_repeat05/epoch_020.pt

# ===============================

python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_TCGA/baseline1_brats21_source/epoch_020.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_TCGA/baseline2_brats21_target_train/epoch_060.pt

python -u eval/evaluate_2d_to_3d.py \
    --config configs_TCGA/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_TCGA/baseline3_brats21_source_plus_target/epoch_020.pt

# ===============================


# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_C4/repeat02/train_config_BraTS2021_Target_train.yaml \
#     --checkpoint outputs_C4_new/target_only_repeat02/epoch_020.pt

# python -u eval/evaluate_2d_to_3d.py \
#     --config configs_C4/repeat02/train_config_BraTS2021_Target_train.yaml \
#     --checkpoint outputs_C4_new/target_only_repeat02/epoch_007.pt

echo "====== All C4 Baseline Evaluations Finished ======"
