"""
E4 loss — combined Dice + BCE for 3D, 2-channel independent-sigmoid (R,L),
mirroring the project's Phase-1 CombinedLoss (loss_function.py 0.5*Dice+0.5*BCE)
but for 5D tensors [B, C, D, H, W]. Sigmoid, not softmax: left/right parotids are
independent binary problems (master doc 10.1).
"""
import torch
import torch.nn as nn


class DiceBCELoss3D(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dw, self.bw, self.smooth = dice_weight, bce_weight, smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        # logits/target: [B, C, D, H, W]
        probs = torch.sigmoid(logits)
        # per-channel global Dice over batch+spatial (dims 0,2,3,4), then mean over C
        dims = (0, 2, 3, 4)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()
        bce_loss = self.bce(logits, target)
        total = self.dw * dice_loss + self.bw * bce_loss
        return total, dice_loss, bce_loss
