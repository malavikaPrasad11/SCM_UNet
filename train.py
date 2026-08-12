"""
Training script for SCM-UNet.

Reproduces the training recipe described in Section 4.2:
  - AdamW optimizer, initial LR 1e-3, weight decay 0.01
  - Cosine Annealing LR schedule
  - BceDiceLoss (Section 3.5)
  - 300 epochs, batch size 32 (defaults; override via CLI)
  - random horizontal/vertical flips + rotations for augmentation
  - fixed global seed (42) for reproducibility

Example:
    python train.py --data_root /path/to/ISIC2018 --img_size 256 \
        --epochs 300 --batch_size 32 --base_dim 64 --out_dir runs/isic18
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_mamba_unet
from dataset import build_datasets
from losses import BceDiceLoss
from metrics import ConfusionMeter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    p = argparse.ArgumentParser(description='Train Mamba-UNet')
    p.add_argument('--data_root', type=str, required=True,
                    help='Dataset root (see dataset.py for expected layout)')
    p.add_argument('--out_dir', type=str, default='runs/scm_unet')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--num_classes', type=int, default=1)
    p.add_argument('--base_dim', type=int, default=64)
    p.add_argument('--d_state', type=int, default=16)
    p.add_argument('--patch_size', type=int, default=4)

    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true', help='use mixed precision')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--log_every', type=int, default=20)
    return p.parse_args()


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    meter = ConfusionMeter()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            meter.update(preds, masks)
    return meter.compute()


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_ds, val_ds, test_ds = build_datasets(
        args.data_root, img_size=args.img_size,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    print(f'train={len(train_ds)} '
          f'val={len(val_ds) if val_ds is not None else 0} '
          f'test={len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               drop_last=True)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_mamba_unet(num_classes=args.num_classes, base_dim=args.base_dim,
                              patch_size=args.patch_size, d_state=args.d_state).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params / 1e6:.2f}M')

    criterion = BceDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(amp_device, enabled=args.amp)

    start_epoch = 0
    best_miou = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', -1.0)
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    eval_loader = val_loader if val_loader is not None else test_loader

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if (step + 1) % args.log_every == 0:
                avg = running_loss / (step + 1)
                print(f'epoch {epoch+1}/{args.epochs} step {step+1}/{len(train_loader)} '
                      f'loss {avg:.4f} lr {scheduler.get_last_lr()[0]:.6f}')

        scheduler.step()
        epoch_loss = running_loss / max(1, len(train_loader))
        dt = time.time() - t0
        print(f'== epoch {epoch+1} done in {dt:.1f}s, train_loss={epoch_loss:.4f} ==')

        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate(model, eval_loader, device)
            print(f'  eval: mIoU={metrics["mIoU"]:.2f} DSC={metrics["DSC"]:.2f} '
                  f'Acc={metrics["Acc"]:.2f} Sen={metrics["Sen"]:.2f} Spe={metrics["Spe"]:.2f}')

            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_miou': best_miou,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'last.pth'))

            if metrics['mIoU'] > best_miou:
                best_miou = metrics['mIoU']
                ckpt['best_miou'] = best_miou
                torch.save(ckpt, os.path.join(args.out_dir, 'best.pth'))
                print(f'  new best mIoU: {best_miou:.2f}, checkpoint saved.')

    print('Training complete. Running final evaluation on test set...')
    best_path = os.path.join(args.out_dir, 'best.pth')
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    test_metrics = evaluate(model, test_loader, device)
    print(f'Test: mIoU={test_metrics["mIoU"]:.2f} DSC={test_metrics["DSC"]:.2f} '
          f'Acc={test_metrics["Acc"]:.2f} Sen={test_metrics["Sen"]:.2f} '
          f'Spe={test_metrics["Spe"]:.2f}')


if __name__ == '__main__':
    main()
