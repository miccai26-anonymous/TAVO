# TAVO: Target-Aware Adaptive Multi-Criterion Data Valuation

Code for "Target-Aware Adaptive Multi-Criterion Data Valuation for Medical Image Segmentation" (anonymous MICCAI 2026 submission).

## Structure

- `Brats2021/` — BraTS brain tumor segmentation experiments
  - `models/` — Model definitions (EfficientViT segmentation)
  - `eval/` — 2D-to-3D evaluation
  - `configs/` — Training configuration templates
  - `scripts/` — Data valuation, CMA-ES search, training, utilities
  - `requirements.txt` — BraTS-specific dependencies
- `mamamia/` — MAMA-MIA breast DCE-MRI experiments
  - `selection/` — Data valuation methods
  - `scripts/` — Embedding extraction, preprocessing, selection, meta-optimization
  - `utils/` — Data loading and metrics
  - `train_seg.py` — EfficientViT training
  - `evaluate_3d_dice.py`, `evaluate_3d_dice_nnunet_preproc.py` — 3D Dice evaluation

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA

```bash
conda create -n tavo python=3.10 -y
conda activate tavo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn scikit-image tqdm matplotlib
pip install cma submodlib_py
pip install SimpleITK
```

For BraTS experiments, see `Brats2021/requirements.txt` for the full dependency list.

### nnUNet (MAMA-MIA)

For the MAMA-MIA experiments, we use the nnUNet fork from [MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA) which provides a 2D PlainConvUNet for breast tumor segmentation.

```bash
git clone https://github.com/LidiaGarrucho/MAMA-MIA externals/MAMA-MIA
pip install -e externals/MAMA-MIA/nnUNet

export nnUNet_raw="$PWD/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_raw"
export nnUNet_preprocessed="$PWD/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_preprocessed"
export nnUNet_results="$PWD/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_results"
```

### EfficientViT

We use [EfficientViT-L1](https://github.com/mit-han-lab/efficientvit) pretrained on ADE20K for semantic segmentation.

```bash
pip install efficientvit
```

## Datasets

### BraTS 2021

The BraTS dataset is available from the [BraTS Challenge](https://www.synapse.org/brats).

### MAMA-MIA

The MAMA-MIA dataset is hosted on [Synapse](https://www.synapse.org/Synapse:syn60868042).

```bash
pip install synapseclient
synapse login --authToken YOUR_TOKEN
synapse get -r syn60868042 --downloadLocation ./data/dataset_mamamia
```

## Usage

### Data Valuation

Compute valuation scores for each selection method:

```bash
# MAMA-MIA
python mamamia/scripts/run_selection.py \
    --method rds \
    --pool-embeddings data/embeddings/pool_embeddings.jsonl \
    --query-embeddings data/embeddings/query_embeddings.jsonl \
    --budget 250 --output outputs/selections/rds_250.json
```

### Meta-Optimization (CMA-ES)

Run CMA-ES to learn optimal combination weights over valuation methods:

```bash
# MAMA-MIA
python mamamia/scripts/run_meta_cmaes.py \
    --pool-embeddings data/embeddings/pool_embeddings.jsonl \
    --query-embeddings data/embeddings/query_embeddings.jsonl \
    --data-dirs /path/to/nnUNet_preprocessed/DatasetXXX/nnUNetPlans_2d \
    --val-cases CASE_A CASE_B \
    --budget 250 --generations 20 --popsize 8
```

### Training and Evaluation

```bash
# MAMA-MIA — EfficientViT training
python mamamia/train_seg.py --help

# MAMA-MIA — 3D Dice evaluation
python mamamia/evaluate_3d_dice_nnunet_preproc.py --help
```

All scripts support `--help` for full usage information.

## Notes

- No private data is included.
- Paths and identifiers are configurable via command-line arguments.
