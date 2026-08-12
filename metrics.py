"""Segmentation metrics matching Table 1/2 of the paper: mIoU, DSC, Acc,
Spe (specificity), Sen (sensitivity/recall).

All functions take `preds` and `targets` as binary {0,1} tensors of the
same shape (any shape works; everything is flattened internally). A running
`ConfusionMeter` is provided for accumulating metrics over a full test set.
"""

import torch


def _confusion(preds, targets, eps=1e-7):
    preds = preds.reshape(-1).float()
    targets = targets.reshape(-1).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    return tp, fp, fn, tn


def compute_metrics(preds, targets, eps=1e-7):
    """Compute mIoU, DSC, Acc, Sen, Spe for one batch (binary segmentation).

    preds/targets: binary tensors, any shape, same shape.
    Returns a dict of Python floats (already *100 as percentages).
    """
    tp, fp, fn, tn = _confusion(preds, targets, eps)

    iou = tp / (tp + fp + fn + eps)
    dsc = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    sen = tp / (tp + fn + eps)  # recall / sensitivity
    spe = tn / (tn + fp + eps)  # specificity

    return {
        'mIoU': iou.item() * 100,
        'DSC': dsc.item() * 100,
        'Acc': acc.item() * 100,
        'Sen': sen.item() * 100,
        'Spe': spe.item() * 100,
    }


class ConfusionMeter:
    """Accumulates TP/FP/FN/TN across many batches, then reports metrics
    computed from the *aggregated* confusion counts (standard practice for
    dataset-level segmentation metrics)."""

    def __init__(self, eps=1e-7):
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, preds, targets):
        tp, fp, fn, tn = _confusion(preds, targets, self.eps)
        self.tp += tp.item()
        self.fp += fp.item()
        self.fn += fn.item()
        self.tn += tn.item()

    def compute(self):
        eps = self.eps
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou = tp / (tp + fp + fn + eps)
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        acc = (tp + tn) / (tp + fp + fn + tn + eps)
        sen = tp / (tp + fn + eps)
        spe = tn / (tn + fp + eps)
        return {
            'mIoU': iou * 100,
            'DSC': dsc * 100,
            'Acc': acc * 100,
            'Sen': sen * 100,
            'Spe': spe * 100,
        }
