# SCM-UNet — From-Scratch PyTorch Reimplementation

An independent, from-scratch PyTorch reimplementation of **SCM-UNet:
Spatial-channel Mamba UNet for medical image segmentation** by Yan et al.,
*Digital Signal Processing*, 168 (2026), 105550.

This implementation was developed directly from the architecture description,
equations, and figures presented in the paper. It does not use or copy the
authors' original implementation.

> **Current experimental status:**  
> The implementation has been validated on a small **500-image subset of
> ISIC2018** using a 350/50/100 train/validation/test split. The current
> results are intended to verify the implementation and training pipeline,
> rather than serve as a full-dataset reproduction of the paper's reported
> results.

---

## Overview

SCM-UNet combines Mamba-based state-space modeling with a U-Net-style
encoder-decoder architecture for medical image segmentation.

The implementation includes:

- 2D Selective Scan (SS2D)
- S6 selective state-space blocks
- Four-directional cross scanning
- VSSLayer encoder/decoder blocks
- Spatial-Channel Attention Bridge (SCAB)
- KAN-based bottleneck
- U-Net encoder-decoder structure
- Skip connections
- BCE + Dice loss
- Standard segmentation metrics
- ISIC2018 dataset loading and deterministic splitting

The Mamba/SSM components are implemented directly in PyTorch rather than
depending on the `mamba_ssm` CUDA package.

---

# Architecture

The main architecture is implemented in `model.py`.

### Main components

| Paper Component | Implementation | Description |
|---|---|---|
| SS2D / Algorithm 1 | `SS2D` | Four-directional 2D selective scan |
| S6 selective scan | `S6Block`, `selective_scan` | State-space sequence processing |
| VSSLayer | `VSSLayer` | Mamba/SS2D branch + gated convolution branch |
| Patch embedding | `PatchEmbed` | Converts input image into feature representation |
| Downsampling | `DownsamplingLayer` | Reduces spatial resolution |
| Upsampling | `UpsamplingLayer` | Recovers spatial resolution |
| Final upsampling | `FinalUpsampling` | Produces full-resolution segmentation output |
| SC-Attention Bridge | `SCAttBridge` | Combines spatial and channel attention |
| Spatial attention | `SpatialAttentionBridge` | Avg/max pooling followed by dilated convolution |
| Channel attention | `ChannelAttentionBridge` | Cross-stage channel attention |
| KAN bottleneck | `KANLinear`, `KANBottleneck` | B-spline-based nonlinear transformation |
| Full network | `SCM_UNet` | Complete encoder-decoder segmentation network |

---

# Repository Structure

```text
SCM_UNet/
│
├── model.py
│   # SCM-UNet architecture
│   # SS2D, S6Block, VSSLayer, SC-Attention Bridge,
│   # KAN bottleneck and encoder-decoder
│
├── dataset.py
│   # Dataset loading, preprocessing and deterministic
│   # train/validation/test splitting
│
├── losses.py
│   # BCE + Dice loss
│
├── metrics.py
│   # mIoU, DSC, Accuracy, Sensitivity and Specificity
│
├── train.py
│   # Main training script
│
├── eval.py
│   # Standalone checkpoint evaluation and inference
│
├── requirements.txt
│   # Python dependencies
│
└── README.md
    # Project documentation
```

---

# Requirements

The implementation requires:

```text
torch
numpy
Pillow
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

The selective scan is implemented using native PyTorch operations, so the
project does not require:

```text
mamba_ssm
causal-conv1d
```

or custom CUDA extensions.

---

# Dataset

## ISIC2018

The current experiment uses the **ISIC2018 Task 1 skin lesion segmentation
dataset**.

The original dataset contains substantially more images than the subset used
in the current experiment. For development and computational feasibility,
only **500 images** were selected for the current training experiment.

### Current split

```text
Total images:       500

Training:           350
Validation:          50
Testing:            100
```

The split is deterministic and controlled by a fixed random seed.

```text
seed = 42
```

---

# Dataset Organization

The dataset is converted into the following structure:

```text
ISIC2018_small/
│
├── images/
│   ├── ISIC_....jpg
│   ├── ISIC_....jpg
│   └── ...
│
└── masks/
    ├── ISIC_....png
    ├── ISIC_....png
    └── ...
```

The dataset loader automatically matches segmentation masks to images.

ISIC2018 masks use the `_segmentation` suffix, for example:

```text
Image:
ISIC_0000001.jpg

Mask:
ISIC_0000001_segmentation.png
```

`dataset.py` supports this naming convention automatically.

---

# Preprocessing

Each image is:

1. Converted to RGB.
2. Resized to `256 × 256`.
3. Normalized to `[0, 1]`.
4. Normalized using ImageNet mean and standard deviation.

The segmentation masks are:

1. Converted to grayscale.
2. Resized using nearest-neighbor interpolation.
3. Converted into binary masks.

The current model therefore receives:

```text
Input:
3 × 256 × 256

Output:
1 × 256 × 256
```

---

# Data Augmentation

The training dataset uses the following augmentations:

* Random horizontal flip
* Random vertical flip
* Random 90-degree rotation

Validation and test images are not augmented.

---

# Training

The main training entry point is:

```bash
python train.py
```

Example:

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

The implementation supports longer training runs through the
`--epochs` argument.

For example:

```bash
--epochs 20
```

or:

```bash
--epochs 300
```

The 300-epoch value follows the training configuration described in the
paper, while the current 500-image experiment uses fewer epochs for
development and validation of the implementation.

---

# Training Configuration

The current small-scale ISIC2018 experiment uses:

| Parameter             |            Value |
| --------------------- | ---------------: |
| Dataset               |         ISIC2018 |
| Number of images      |              500 |
| Training images       |              350 |
| Validation images     |               50 |
| Test images           |              100 |
| Image size            |        256 × 256 |
| Base dimension        |               64 |
| SSM state dimension   |               16 |
| Initial learning rate |             1e-3 |
| Weight decay          |             0.01 |
| Optimizer             |            AdamW |
| LR scheduler          | Cosine Annealing |
| Loss                  |       BCE + Dice |
| Random seed           |               42 |
| GPU                   |  NVIDIA Tesla T4 |

---

# Loss Function

The implementation uses the BCE + Dice loss described in the paper:

```text
L = 0.5 × BCE + 0.5 × DiceLoss
```

The implementation is provided in:

```text
losses.py
```

---

# Checkpoints

During training, two checkpoints are produced:

```text
runs/isic18_small/
├── last.pth
└── best.pth
```

### `last.pth`

Contains the latest training state.

### `best.pth`

Contains the model checkpoint with the best validation mIoU observed during
training.

The checkpoint also stores:

* Model state
* Optimizer state
* Scheduler state
* Epoch
* Best mIoU
* Training arguments

---

# Evaluation

After training, the trained checkpoint can be evaluated using:

```bash
python eval.py \
    --data_root /path/to/ISIC2018_small \
    --ckpt runs/isic18_small/best.pth \
    --img_size 256 \
    --batch_size 16 \
    --threshold 0.5
```

Predicted segmentation masks can optionally be saved:

```bash
python eval.py \
    --data_root /path/to/ISIC2018_small \
    --ckpt runs/isic18_small/best.pth \
    --img_size 256 \
    --batch_size 16 \
    --threshold 0.5 \
    --save_preds preds/isic18_small
```

The evaluation script reports:

* mIoU
* DSC
* Accuracy
* Sensitivity
* Specificity

---

# Current Experimental Results

The current implementation was evaluated on the **100-image test split** of
the 500-image ISIC2018 subset.

### Test Results

| Metric          |     Result |
| --------------- | ---------: |
| **mIoU**        | **72.94%** |
| **DSC**         | **84.35%** |
| **Accuracy**    | **93.21%** |
| **Sensitivity** | **78.35%** |
| **Specificity** | **97.74%** |

These results are from the current small-scale experiment and should not be
interpreted as a full-dataset reproduction of the results reported in the
original paper.

---

# Selective Scan Implementation

The SSM selective scan is implemented directly in PyTorch.

The implementation uses a vectorized scan formulation with cumulative
operations in log-space.

This avoids requiring:

```text
mamba_ssm
causal-conv1d
custom CUDA kernels
```

and allows the implementation to run on both CPU and GPU.

The trade-off is that the pure PyTorch implementation can require more
memory and may be slower than optimized CUDA implementations for large
inputs.

---

# Four-Directional SS2D

The SS2D implementation performs four directional scans:

```text
→
←
↓
↑
```

These are implemented through row-major and column-major ordering together
with their reverse directions.

The resulting directional features are combined to provide 2D spatial
context.

---

# VSSLayer

The VSSLayer contains two main branches:

```text
Input
  │
  ├── SS2D / S6 branch
  │
  └── Gated convolution branch
          │
          ↓
       Fusion
          │
       Residual
          │
        Output
```

This follows the dual-branch structure described for the VSSLayer in the
SCM-UNet architecture.

---

# SC-Attention Bridge

The Spatial-Channel Attention Bridge is implemented through:

```text
SpatialAttentionBridge
ChannelAttentionBridge
SCAttBridge
```

### Spatial attention

The spatial branch uses:

```text
Average Pooling
      +
Max Pooling
      ↓
Dilated 7×7 Convolution
      ↓
Spatial Attention
```

### Channel attention

The channel branch uses global feature aggregation followed by channel-wise
gating.

The bridge is used to enhance the information passed through the encoder
skip connections.

---

# KAN Bottleneck

The bottleneck contains:

```text
KANLinear
KANBottleneck
```

The KAN implementation uses learnable B-spline basis functions to provide
nonlinear transformations at the bottleneck.

This corresponds to the KAN-based transformation described in the SCM-UNet
paper.

---

# Encoder-Decoder Structure

The network follows a U-Net-style encoder-decoder architecture.

The encoder contains four VSSLayer stages.

The bottleneck contains the KAN transformation.

The decoder reconstructs the spatial resolution using upsampling layers and
VSSLayer blocks.

The implementation uses three primary skip connections through the
SC-Attention Bridge, followed by an additional decoder refinement stage.

The exact stage/skip interpretation is documented in the `SCM_UNet` class
implementation because the architecture figure in the paper leaves some
stage/skip details open to interpretation.

---

# Important Implementation Notes

## From-scratch implementation

The model architecture was implemented independently rather than importing
an existing Mamba-UNet implementation.

The main Mamba/SSM-related components are implemented in:

```text
model.py
```

including:

```text
SS2D
S6Block
selective_scan
VSSLayer
```

The complete SCM-UNet architecture is assembled in the same file.

---

## Difference from optimized Mamba implementations

This implementation does not depend on the official/optimized Mamba CUDA
selective-scan implementation.

Instead, it uses native PyTorch operations.

Therefore:

* It is easier to run in environments such as Kaggle.
* It does not require compiling custom CUDA extensions.
* It can run without `mamba_ssm`.
* It may use more GPU memory.
* It may be slower than optimized Mamba implementations.

---

# Current Scope and Limitations

The current experiment is intended primarily to validate the implementation
and training pipeline.

### Current limitations

* Only a **500-image subset** of ISIC2018 is currently used.
* The test set contains **100 images**.
* The reported metrics therefore cannot be directly compared with full
  dataset results from the original paper.
* Training is currently being performed with a reduced number of epochs for
  experimentation.
* The pure PyTorch selective scan is less optimized than a custom CUDA
  implementation.

A full reproduction should train on the complete dataset using the training
configuration described in the paper and should follow the exact dataset
split used by the paper.

---

# Quick Model Sanity Check

To test whether the architecture can be instantiated independently of a
dataset:

```bash
python model.py
```

This performs a forward pass using a random input tensor and reports the
output shape and model parameter count.

---

# Reproducibility

A fixed global seed is used:

```text
42
```

The dataset split, random augmentation operations, NumPy operations and
PyTorch operations are initialized using the same seed.

The current experiment therefore uses the deterministic split:

```text
500 images
   ↓
350 train
50 validation
100 test
```

---

# Citation

If this implementation is used in academic work, cite the original SCM-UNet
paper:

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

---

# Project Status

### Implementation

* [x] PyTorch implementation
* [x] S6 selective scan
* [x] Four-directional SS2D
* [x] VSSLayer
* [x] U-Net encoder-decoder
* [x] SC-Attention Bridge
* [x] KAN bottleneck
* [x] BCE + Dice loss
* [x] Segmentation metrics
* [x] ISIC2018 dataset pipeline
* [x] Training pipeline
* [x] Checkpoint saving
* [x] Standalone evaluation
* [x] Predicted mask generation

### Experimental validation

* [x] 500-image ISIC2018 subset
* [x] 350/50/100 train/validation/test split
* [x] GPU training on Tesla T4
* [x] Checkpoint-based test evaluation
* [x] Test mIoU: **72.94%**
* [x] Test DSC: **84.35%**

### Future work

* [ ] Train on the complete ISIC2018 dataset
* [ ] Run longer training following the paper configuration
* [ ] Compare against baseline segmentation models
* [ ] Generate qualitative prediction visualizations
* [ ] Evaluate on additional benchmark datasets
* [ ] Perform ablation studies for SCM components
