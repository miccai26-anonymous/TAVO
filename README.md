# Target-Aware Adaptive Multi-Criterion Data Valuation

Code for "Target-Aware Adaptive Multi-Criterion Data Valuation for Medical Image Segmentation" (anonymous MICCAI 2026 submission).

## Structure

- `eval/` — Evaluation scripts
- `models/` — Model definitions (EfficientViT segmentation)
- `scripts/` — BraTS experiments (selection, CMA-ES search, training)
- `mamamia/` — MAMA-MIA breast DCE-MRI experiments (selection, meta-optimization, training, evaluation)

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA

```bash
conda create -n data_selection python=3.10 -y
conda activate data_selection

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn scikit-image tqdm matplotlib
pip install cma botorch gpytorch
pip install SimpleITK synapseclient
```

### nnUNet (MAMA-MIA fork)

We use the nnUNet fork from [MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA), which includes a 2D PlainConvUNet for breast tumor segmentation.

```bash
git clone https://github.com/LidiaGarrucho/MAMA-MIA externals/MAMA-MIA
pip install -e externals/MAMA-MIA/nnUNet

export nnUNet_raw="$PWD/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_raw"
export nnUNet_preprocessed="$PWD/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_preprocessed"
export nnUNet_results="$PWD/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_results"
```

### EfficientViT

We use [EfficientViT-L1](https://github.com/mit-han-lab/efficientvit) pretrained on ADE20K (150-class semantic segmentation). The 150-class head is kept during fine-tuning — label 0 is background, label >0 is tumor.

```bash
git clone https://github.com/mit-han-lab/efficientvit models/efficientvit
pip install -e models/efficientvit

mkdir -p models/efficientvit/assets/checkpoints/efficientvit_seg
wget -O models/efficientvit/assets/checkpoints/efficientvit_seg/efficientvit_seg_l1_ade20k.pt \
    https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_ade20k.pt
```

## Datasets

### BraTS (Brain Tumor Segmentation)

The BraTS dataset is available from the [BraTS Challenge](https://www.synapse.org/brats).

### MAMA-MIA (Breast DCE-MRI)

The MAMA-MIA dataset is hosted on [Synapse](https://www.synapse.org/Synapse:syn60868042). You need a Synapse account.

```bash
pip install synapseclient
synapse login --authToken YOUR_TOKEN
synapse get -r syn60868042 --downloadLocation ./data/dataset_mamamia
```

After downloading, the data should be at `data/dataset_mamamia/` with:
- `images/` — per-case folders with multi-phase NIfTI files
- `segmentations/expert/` — ground truth labels

## Quick Start

1. Install dependencies and download models (see Setup above)
2. Download dataset(s)
3. Update dataset paths in scripts
4. Run training/evaluation scripts

All scripts support `--help` for full usage information.

## Notes

- No private data is included.
- Paths and identifiers are configurable via command-line arguments.
