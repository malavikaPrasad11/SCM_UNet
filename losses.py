"""Loss functions used to train SCM-UNet (Section 3.5 / Eq. 21-23)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BceDiceLoss(nn.Module):
    """L = 0.5 * BCE + 0.5 * Dice, computed on sigmoid probabilities.

    Expects `logits` of shape (B, 1, H, W) and `targets` of shape
    (B, 1, H, W) with values in {0, 1}.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = 1 - (2 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = dice.mean()

        return (bce_loss + dice_loss) / 2.0
