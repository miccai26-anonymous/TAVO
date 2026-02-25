#!/bin/bash
#SBATCH --job-name=Eval_ORIENT
#SBATCH --output=logs/eval_orient_%j.out
#SBATCH --error=logs/eval_orient_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
ulimit -n 4096

source ~/envs/brain310/bin/activate
export PYTHONPATH=.

cd /path/to/project

echo "====== Running ORIENT Evaluations ======"
CONFIG="configs_C4/train_config_BraTS21_orient_15T.yaml"
   
CKPT="outputs_C4/Brats21_S_orient_repeat01_epoch3_15T_plus_T_train/best.pt"
   
python -u eval/evaluate_2d_to_3d.py \
    --config $CONFIG \
    --checkpoint $CKPT
echo "====== ORIENT Evaluation Completed ======"
