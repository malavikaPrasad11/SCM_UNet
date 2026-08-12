# SCM-UNet (reimplementation)

An independent, from-scratch PyTorch reimplementation of **SCM-UNet:
Spatial-channel Mamba UNet for medical image segmentation** (Yan et al.,
*Digital Signal Processing* 168 (2026) 105550), built directly from the
architecture description, equations, and figures in the paper. It does not
use or copy the authors' original code.

## What's implemented

| Paper component | File | Notes |
|---|---|---|
| SS2D (Algorithm 1) / S6 selective scan | `model.py: SS2D, S6Block, selective_scan` | 4-directional cross-scan, fully vectorized (parallel, log-space) selective-scan — no custom CUDA kernel needed, runs on CPU or GPU. |
| VSSLayer (Fig. 3) | `model.py: VSSLayer` | Dual-branch (SS2D branch + gated conv branch) with residual connection. |
| Encoder/decoder + ImageTokenizer/Up-Down sampling (Fig. 2) | `model.py: PatchEmbed, DownsamplingLayer, UpsamplingLayer, FinalUpsampling, SCM_UNet` | 4 encoder VSSLayer stages, KAN bottleneck, 3 skip connections + 1 refinement decoder stage (see docstring in `SCM_UNet` for the exact topology choice, since the figure is slightly ambiguous about stage/skip counts). |
| SC-Attention Bridge / SCAB (Fig. 5, Eq. 9-16) | `model.py: SpatialAttentionBridge, ChannelAttentionBridge, SCAttBridge` | Shared-weight spatial attention (avg+max pool → dilated 7×7 conv) followed by cross-stage channel attention (GAP → 1D conv → per-stage FC gates). |
| KANLinear bottleneck (Fig. 6, Eq. 17-20) | `model.py: KANLinear, KANBottleneck` | Learnable B-spline activation function per input-output connection (standard "efficient-KAN" formulation), used at the network bottleneck. |
| BceDiceLoss (Eq. 21-23) | `losses.py` | 0.5·BCE + 0.5·Dice. |
| mIoU / DSC / Acc / Sen / Spe metrics | `metrics.py` | Batch-level and dataset-level (`ConfusionMeter`) variants, matching Table 1/2 of the paper. |
| Dataset loading (ISIC17/18, Kvasir-SEG, ColonDB, BUSI) | `dataset.py` | Generic binary-mask segmentation dataset with flexible folder layout + deterministic splitting. |
| Training loop (Section 4.2) | `train.py` | AdamW, cosine annealing LR, 300 epochs / batch size 32 defaults, flips + rotation augmentation, fixed seed 42. |
| Evaluation / inference | `eval.py` | Loads a checkpoint, reports test-set metrics, optionally dumps predicted masks. |

## Repository layout

```
.
├── model.py         # SCM-UNet architecture (SS2D, VSSLayer, SC-Att Bridge, KANLinear, full model)
├── dataset.py       # Segmentation dataset + train/val/test split utilities
├── losses.py        # BceDiceLoss
├── metrics.py       # mIoU / DSC / Acc / Sen / Spe
├── train.py         # Training entry point
├── eval.py          # Standalone evaluation / inference entry point
├── README.md        # This documentation
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Only `torch`, `numpy`, and `Pillow` are required — no `mamba_ssm` /
`causal-conv1d` CUDA extensions, since the selective scan is implemented
natively in PyTorch (see "Implementation notes" below).

## Data preparation

Each dataset (ISIC2017, ISIC2018, Kvasir-SEG, ColonDB, BUSI) should be
organized as a binary-mask segmentation folder. Two layouts are supported
by `dataset.py`:

**Option A — pre-made splits:**

```
<data_root>/
├── train/
│   ├── images/  *.png|*.jpg
│   └── masks/   <same-stem>.png   (0/255 binary masks)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Option B — flat folder, auto-split:**

```
<data_root>/
├── images/
└── masks/
```

If `<data_root>/train/images` doesn't exist, `train.py`/`eval.py` will
deterministically split `<data_root>/images` into train/val/test using
`--val_ratio` / `--test_ratio` and `--seed` (default seed 42, matching the
paper).

Mask files are matched to images by filename stem, with a few common
suffix fallbacks (`_segmentation`, `_mask`, `_gt`) to support ISIC-style
naming.

## Training

```bash
python train.py \
    --data_root /path/to/ISIC2018 \
    --img_size 256 \
    --epochs 300 \
    --batch_size 32 \
    --base_dim 64 \
    --d_state 16 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --out_dir runs/isic18
```

Key flags:

- `--base_dim`: channel width of the first encoder stage (paper-scale ≈ 64).
  Reduce this (e.g. 16–32) for quick experiments or memory-constrained
  machines — the model's parameter count and activation memory scale
  roughly with `base_dim`.
- `--d_state`: SSM state dimension per channel (paper-style Mamba default
  16).
- `--img_size`: input resolution (paper uses 256×256 for all five
  datasets).
- `--amp`: enable mixed-precision training on CUDA.
- `--resume path/to/ckpt.pth`: resume training from a checkpoint.

Checkpoints (`last.pth`, `best.pth`) and training logs are written to
`--out_dir`. `best.pth` is the checkpoint with the highest validation (or
test, if no val split exists) mIoU seen so far.

## Evaluation

```bash
python eval.py \
    --data_root /path/to/ISIC2018 \
    --ckpt runs/isic18/best.pth \
    --img_size 256 \
    --save_preds preds/isic18   # optional: dump predicted masks as PNGs
```

This prints dataset-level mIoU / DSC / Acc / Sen / Spe (aggregated over the
full confusion matrix, not averaged per-image), matching the metrics
reported in Table 1/2 of the paper.

## Quick sanity check (no dataset needed)

```bash
python model.py
```

Runs a forward pass on a random tensor and prints the output shape and
parameter count, useful for confirming the environment/install is correct
before starting real training.

## Implementation notes / deviations from the paper

- **Selective scan**: implemented as a fully vectorized parallel scan in
  pure PyTorch using cumulative sums in log-space, rather than the
  `mamba_ssm` CUDA kernel used in most official Mamba/VMamba
  implementations. This makes the code portable (CPU or GPU, no custom
  kernel build step) at some cost in training throughput on large images —
  for large-scale reproduction of the paper's exact wall-clock numbers you
  may want to swap in `mamba_ssm`'s CUDA `selective_scan_fn` as a drop-in
  replacement for `selective_scan()` in `model.py`.
- **4-direction scan**: implemented via row-major / column-major orderings
  and their reverses (the standard VMamba-style cross-scan), which realize
  the four corner-to-corner orientations shown in Fig. 4.
- **Encoder/decoder skip topology**: Fig. 2 shows 4 encoder `VSSLayer`
  stages but only 3 `SC_Att_Bridge` skip arrows (stages 1-3), with stage 4
  feeding directly into the `KANLinear` bottleneck. This reimplementation
  follows that reading literally: 3 skip connections with matching
  channel/resolution pairing, plus one extra decoder refinement `VSSLayer`
  (no skip) before the `FinalUpsampling` head, so that there are still 4
  decoder `VSSLayer_up` blocks as drawn. This choice is documented in the
  `SCM_UNet` class docstring in `model.py`.
- **KANLinear**: implemented following the standard "efficient-KAN"
  B-spline formulation (Liu et al., 2024), applied at the bottleneck as
  `LN(Z + Conv(Φ(Z)))` per Eq. 17.
- Loss, optimizer, LR schedule, augmentation, and dataset-level metrics
  follow Sections 3.5 and 4.2 as closely as the paper text specifies.

## Citation

If you use this code, please cite the original paper:

```
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
