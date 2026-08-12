# SCM-UNet — From-Scratch PyTorch Reimplementation

An independent, from-scratch PyTorch reimplementation of **SCM-UNet: Spatial-channel Mamba UNet for medical image segmentation** by Yan et al., *Digital Signal Processing*, 168 (2026), 105550.

Built directly from the paper's architecture description and equations — not copied from the authors' code.

> **Status:** Validated on a small **500-image subset of ISIC2018** (350 train / 50 val / 100 test). This is a small-scale sanity check of the implementation and training pipeline, **not** a full-dataset reproduction of the paper's benchmark results.

---

## What's implemented

- 2D Selective Scan (SS2D), four-directional cross scanning
- S6 selective state-space blocks
- VSSLayer encoder/decoder blocks
- Spatial-Channel Attention Bridge (SCAB)
- KAN-based bottleneck
- U-Net encoder-decoder with skip connections
- BCE + Dice loss, standard segmentation metrics

The Mamba/SSM components are pure PyTorch — no `mamba_ssm` or custom CUDA kernels required, so it runs anywhere (including CPU).

---

## Repo structure

```text
model.py      # SCM-UNet architecture (SS2D, S6Block, VSSLayer, SCAB, KAN bottleneck)
dataset.py    # Dataset loading + deterministic train/val/test split
losses.py     # BCE + Dice loss
metrics.py    # mIoU, DSC, Accuracy, Sensitivity, Specificity
train.py      # Training script
eval.py       # Checkpoint evaluation / inference
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires only `torch`, `numpy`, `Pillow`.

---

## Dataset

Uses a 500-image subset of **ISIC2018 Task 1** (skin lesion segmentation), split deterministically (seed=42) into 350 train / 50 val / 100 test. Images are resized to 256×256 and normalized with ImageNet stats; masks are converted to binary.

Expected layout:

```text
ISIC2018_small/
├── images/ISIC_....jpg
└── masks/ISIC_...._segmentation.png
```

---

## Training

```bash
python train.py \
    --data_root /path/to/ISIC2018_small \
    --img_size 256 \
    --epochs 5 \
    --batch_size 2 \
    --base_dim 64 \
    --d_state 16 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --out_dir runs/isic18_small \
    --seed 42
```

The current experiment used 5 epochs on a Tesla T4 (AMP disabled to avoid an earlier NaN issue). The paper's full config uses 300 epochs on the complete dataset — increase `--epochs` for longer runs.

Checkpoints (`last.pth`, `best.pth`) are saved to `--out_dir`.

---

## Evaluation

```bash
python eval.py \
    --data_root /path/to/ISIC2018_small \
    --ckpt runs/isic18_small/best.pth \
    --img_size 256 \
    --batch_size 16 \
    --threshold 0.5
```

Add `--save_preds <dir>` to save predicted masks.

---

## Current results (100-image test split)

| Metric | Result |
|---|---:|
| mIoU | 72.94% |
| DSC | 84.35% |
| Accuracy | 93.21% |
| Sensitivity | 78.35% |
| Specificity | 97.74% |

These numbers are from the 500-image subset only and shouldn't be compared directly against the paper's full-dataset results.

---

## Limitations

- Only 500 of the full ISIC2018 images are used so far
- Trained for a small number of epochs (development run, not the paper's 300-epoch config)
- Pure-PyTorch selective scan is less optimized than a custom CUDA kernel

A full reproduction would train on the complete dataset with the paper's full training configuration.

---

## Citation

```bibtex
@article{yan2026scmunet,
  title   = {SCM-UNet: Spatial-channel Mamba UNet for medical image segmentation},
  author  = {Yan, Haijie and Hong, Qiuhong and Wei, Shoulin and Zhang, Xiangliang and Yin, Jibin},
  journal = {Digital Signal Processing},
  volume  = {168},
  pages   = {105550},
  year    = {2026},
  doi     = {10.1016/j.dsp.2025.105550}
}
```
