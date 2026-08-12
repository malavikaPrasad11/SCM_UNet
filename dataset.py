"""
Dataset loader for the five benchmarks used in the paper (ISIC2017,
ISIC2018, Kvasir-SEG, ColonDB, BUSI). All of them are treated uniformly as
binary segmentation datasets with the following expected directory layout:

    <root>/
        images/   *.png | *.jpg | *.bmp | ...
        masks/    <same-stem>.png  (binary masks, 0=background, 255=foreground)

Train/val/test split is done either by:
  - separate subfolders: <root>/train/images, <root>/train/masks,
                          <root>/val/images,   <root>/val/masks,  ... or
  - a single images/masks folder plus a deterministic random split
    controlled by --val_ratio / --test_ratio and --seed.

This mirrors how the paper's five public datasets are commonly prepared
(e.g. the widely-used ISIC/Kvasir/BUSI split scripts from the VM-UNet /
UltraLight-VM-UNet repos), without depending on any dataset-specific code.
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _list_images(folder):
    files = [f for f in sorted(os.listdir(folder))
             if f.lower().endswith(IMG_EXTS)]
    return files


def _find_mask_path(masks_dir, image_filename):
    stem = os.path.splitext(image_filename)[0]
    for ext in IMG_EXTS:
        cand = os.path.join(masks_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    # some datasets suffix masks, e.g. ISIC "_segmentation"
    for suffix in ('_segmentation', '_mask', '_gt'):
        for ext in IMG_EXTS:
            cand = os.path.join(masks_dir, stem + suffix + ext)
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(f'No mask found for image {image_filename} in {masks_dir}')


class SegmentationDataset(Dataset):
    """Generic binary-mask segmentation dataset.

    Args:
        images_dir, masks_dir: folders containing images and masks.
        file_list: optional explicit list of image filenames to use
            (for pre-computed splits). If None, all images in images_dir
            are used.
        img_size: (H, W) to resize every image/mask to.
        augment: whether to apply random flip/rotation augmentation
            (paper: random horizontal/vertical flips + random rotations).
    """

    def __init__(self, images_dir, masks_dir, file_list=None, img_size=256,
                 augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.augment = augment

        self.files = file_list if file_list is not None else _list_images(images_dir)

    def __len__(self):
        return len(self.files)

    def _load(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.images_dir, fname)
        mask_path = _find_mask_path(self.masks_dir, fname)

        img = Image.open(img_path).convert('RGB').resize(
            self.img_size[::-1], Image.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize(
            self.img_size[::-1], Image.NEAREST)
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        img = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        mask = np.asarray(mask, dtype=np.float32) / 255.0  # (H,W)
        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        mask = torch.from_numpy(mask[None, ...].copy()).float()
        return img, mask

    @staticmethod
    def _augment(img, mask):
        # random horizontal flip
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        # random vertical flip
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        # random 90-degree rotation (cheap, avoids interpolation artifacts)
        k = random.randint(0, 3)
        if k > 0:
            img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
        return img, mask


def make_splits(images_dir, val_ratio=0.1, test_ratio=0.2, seed=42):
    """Deterministically split a flat images/ folder into train/val/test
    filename lists (used only when the dataset root has no pre-made
    train/val/test subfolders)."""
    files = _list_images(images_dir)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_files = files[:n_test]
    val_files = files[n_test:n_test + n_val]
    train_files = files[n_test + n_val:]
    return train_files, val_files, test_files


def build_datasets(root, img_size=256, val_ratio=0.1, test_ratio=0.2, seed=42):
    """Build (train_ds, val_ds, test_ds) from a dataset root.

    Two layouts are supported:
      1) root/train/{images,masks}, root/val/{images,masks},
         root/test/{images,masks}   -> used as-is.
      2) root/images, root/masks    -> deterministically split according to
         val_ratio / test_ratio.
    """
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    test_dir = os.path.join(root, 'test')

    if os.path.isdir(train_dir) and os.path.isdir(os.path.join(train_dir, 'images')):
        train_ds = SegmentationDataset(
            os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'),
            img_size=img_size, augment=True)
        val_ds = None
        if os.path.isdir(val_dir):
            val_ds = SegmentationDataset(
                os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'),
                img_size=img_size, augment=False)
        test_ds = SegmentationDataset(
            os.path.join(test_dir, 'images'), os.path.join(test_dir, 'masks'),
            img_size=img_size, augment=False)
        return train_ds, val_ds, test_ds

    images_dir = os.path.join(root, 'images')
    masks_dir = os.path.join(root, 'masks')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f'Could not find {images_dir}. Expected either '
            f'{root}/train/images (+ val/, test/) or {root}/images (+ masks/).'
        )
    train_files, val_files, test_files = make_splits(
        images_dir, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    train_ds = SegmentationDataset(images_dir, masks_dir, train_files,
                                    img_size=img_size, augment=True)
    val_ds = SegmentationDataset(images_dir, masks_dir, val_files,
                                  img_size=img_size, augment=False) if val_files else None
    test_ds = SegmentationDataset(images_dir, masks_dir, test_files,
                                   img_size=img_size, augment=False)
    return train_ds, val_ds, test_ds
