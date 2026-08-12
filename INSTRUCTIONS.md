# Project Instructions

This repository contains a PyTorch reimplementation of SCM-UNet, a medical image segmentation model. The code is written so that someone familiar with basic Python and machine learning concepts can understand what each file does and how the project is structured.

## What this project does

- Loads medical image datasets and binary segmentation masks.
- Builds a neural network model called SCM-UNet for predicting segmentation masks.
- Trains the model on a dataset.
- Evaluates the trained model and reports segmentation quality metrics.

The model is designed for binary segmentation, where each pixel is classified as either foreground (object) or background.

## Files in this repository

### `dataset.py`

This file handles data loading and preprocessing.

- Reads images and their matching mask files from disk.
- Supports two dataset formats:
  - separate `train`, `val`, and `test` folders each with `images/` and `masks/`
  - a flat `images/` and `masks/` folder with automatic train/validation/test splitting.
- Resizes images and masks to a fixed size.
- Converts masks into binary values (0 or 1).
- Applies simple data augmentation during training: random horizontal flip, random vertical flip, and random rotation by 90 degrees.
- Normalizes image pixel values using standard ImageNet mean and standard deviation.

### `model.py`

This file defines the SCM-UNet neural network architecture.

- Implements the core selective scan block used for efficient sequence processing.
- Builds a 2D selective scan layer that reads image features in four directions.
- Defines a VSSLayer, which combines selective scan processing with other neural network operations.
- Defines patch embedding, downsampling, upsampling, and the final model architecture.
- The final model takes a color image and outputs a single-channel mask prediction.

### `losses.py`

This file defines the training loss used to teach the model.

- Uses `BceDiceLoss`, a combination of binary cross-entropy loss and Dice loss.
- This loss function helps the model learn both pixel-wise accuracy and good overlap with the ground truth mask.

### `metrics.py`

This file computes evaluation metrics for segmentation quality.

- Computes metrics such as:
  - mIoU (mean Intersection over Union)
  - DSC (Dice Similarity Coefficient)
  - Accuracy
  - Sensitivity (recall for the positive class)
  - Specificity (recall for the negative class)
- Maintains a confusion matrix for all predictions and ground truth masks.

### `train.py`

This is the training script.

- Parses command-line options for dataset location, training settings, and model size.
- Loads the dataset and creates data loaders for training, validation, and testing.
- Builds the SCM-UNet model.
- Uses the AdamW optimizer and cosine annealing learning rate schedule.
- Trains for a number of epochs, computing loss on each training batch.
- Evaluates the model periodically on validation or test data.
- Saves checkpoints to disk, including the latest model and the best-performing model based on validation mIoU.

### `eval.py`

This is the evaluation script.

- Loads a saved model checkpoint.
- Builds the test dataset and runs the model on it.
- Computes segmentation metrics on the test set.
- Optionally saves predicted mask images to disk.

### `README.md`

This file contains general project documentation, including:

- A summary of the architecture and implementation.
- Installation instructions.
- Dataset preparation details.
- Training and evaluation command examples.

### `requirements.txt`

This file lists the Python packages needed to run the code.

- Install the dependencies with:

```bash
pip install -r requirements.txt
```

## How to use this project

1. Prepare your dataset.

- If your data already has separate `train`, `val`, and `test` folders, use that layout.
- If your data has only one `images/` folder and one `masks/` folder, the code will split it automatically.

2. Train the model.

```bash
python train.py --data_root /path/to/dataset --img_size 256 --epochs 300 --batch_size 32
```

3. Evaluate the model.

```bash
python eval.py --data_root /path/to/dataset --ckpt runs/scm_unet/best.pth --img_size 256
```

4. Save predicted masks (optional).

```bash
python eval.py --data_root /path/to/dataset --ckpt runs/scm_unet/best.pth --img_size 256 --save_preds preds
```

## Plain English summary of the workflow

1. `dataset.py` reads images and masks, resizes them, normalizes them, and prepares them for training.
2. `model.py` defines the neural network that learns to predict masks from images.
3. `train.py` trains the model using labeled image-mask pairs.
4. `eval.py` loads a trained model and checks how well it predicts masks on new data.
5. `losses.py` defines how the model is penalized when its predictions are wrong.
6. `metrics.py` defines the numbers that show how good the model is.

## Notes

- The model is designed for binary segmentation, so it predicts a single mask channel.
- The code uses PyTorch, a popular machine learning library.
- The dataset must use matching filenames for images and masks, or masks must be named with a known suffix like `_segmentation`, `_mask`, or `_gt`.
