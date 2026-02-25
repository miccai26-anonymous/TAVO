#!/bin/bash
#SBATCH --job-name=Eval_C4_Baselines
#SBATCH --output=logs/C4_new/eval_C4_baselines_%j.out
#SBATCH --error=logs/C4_new/eval_C4_baselines_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
ulimit -n 4096

# ===============================
# Environment
# ===============================
source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "====== Running C4 Baseline Evaluations ======"

# ============================================================
# 1️⃣ Source only (5 repeats)
# ============================================================
# for i in 01 02 03 04 05; do
#   echo ">>> Eval Source Only - repeat ${i}"
#   python -u eval/evaluate_2d_to_3d.py \
#     --config configs_C4/repeat${i}/train_config_BraTS2021_Source.yaml \
#     --checkpoint outputs_C4/source_only_${i}/best.pt
# done
python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4_new/train_config_BraTS2021_Source.yaml \
    --checkpoint outputs_C4_new/source_only_01/best.pt

# ============================================================
# 2️⃣ Target train only (5 repeats)
# ============================================================
# for i in 01 02 03 04 05; do
#   echo ">>> Eval Target Train Only - repeat ${i}"
#   python -u eval/evaluate_2d_to_3d.py \
#     --config configs_C4/repeat${i}/train_config_BraTS2021_Target_train.yaml \
#     --checkpoint outputs_C4/target_only_repeat${i}/best.pt
# done

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4_new/train_config_BraTS2021_Target_train.yaml \
    --checkpoint outputs_C4_new/target_only_repeat01/best.pt

# ============================================================
# 3️⃣ Source + Target train (5 repeats)
# ============================================================
# for i in 01 02 03 04 05; do
#   echo ">>> Eval Source + Target - repeat ${i}"
#   python -u eval/evaluate_2d_to_3d.py \
#     --config configs_C4/repeat${i}/train_config_BraTS2021_S_and_T.yaml \
#     --checkpoint outputs_C4/source_plus_target_repeat${i}/best.pt
# done

python -u eval/evaluate_2d_to_3d.py \
    --config configs_C4_new/train_config_BraTS2021_S_and_T.yaml \
    --checkpoint outputs_C4_new/source_plus_target_repeat01/best.pt

echo "====== All C4 Baseline Evaluations Finished ======"
