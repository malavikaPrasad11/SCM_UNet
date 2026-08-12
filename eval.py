"""
Standalone evaluation / inference script for a trained SCM-UNet checkpoint.

Example:
    python eval.py --data_root /path/to/ISIC2018 --ckpt runs/isic18/best.pth \
        --img_size 256 --save_preds preds/isic18
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import build_scm_unet
from dataset import build_datasets
from metrics import ConfusionMeter


def get_args():
    p = argparse.ArgumentParser(description='Evaluate SCM-UNet')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--save_preds', type=str, default=None,
                    help='If set, save predicted masks (as PNGs) to this directory')
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get('args', {})
    base_dim = train_args.get('base_dim', 64)
    d_state = train_args.get('d_state', 16)
    patch_size = train_args.get('patch_size', 4)
    num_classes = train_args.get('num_classes', 1)

    model = build_scm_unet(num_classes=num_classes, base_dim=base_dim,
                            patch_size=patch_size, d_state=d_state).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    _, _, test_ds = build_datasets(
        args.data_root, img_size=args.img_size,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)

    meter = ConfusionMeter()
    idx = 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > args.threshold).float()
            meter.update(preds, masks)

            if args.save_preds:
                preds_np = (preds.cpu().numpy()[:, 0] * 255).astype(np.uint8)
                for b in range(preds_np.shape[0]):
                    fname = test_ds.files[idx]
                    stem = os.path.splitext(fname)[0]
                    Image.fromarray(preds_np[b]).save(
                        os.path.join(args.save_preds, f'{stem}_pred.png'))
                    idx += 1

    metrics = meter.compute()
    print('Test set results:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.2f}')


if __name__ == '__main__':
    main()
