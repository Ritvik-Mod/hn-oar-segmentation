"""
Masked partial-label loss for multi-organ segmentation.

Independent sigmoid channels + masked (BCE + Dice). For each sample, only the
classes actually annotated in that sample contribute to the loss; unannotated
classes are ignored so the network is never penalised for correctly predicting
an organ the clinician simply did not contour.

Works for 2D ([B,C,H,W]) or 3D ([B,C,D,H,W]) — spatial dims are reduced
dynamically. Adapted from a drafted recipe, with a corrected Dice reduction
(masked classes must not leak into the averaged Dice).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPartialLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets, mask):
        """
        logits, targets: [B, C, *spatial]   (targets in {0,1})
        mask:            [B, C]              (1 if class annotated in that sample)
        """
        spatial = tuple(range(2, logits.dim()))
        probs = torch.sigmoid(logits)
        mask_b = mask.view(mask.shape[0], mask.shape[1], *([1] * len(spatial)))
        valid_classes = mask.sum() + self.smooth  # total annotated (sample,class) pairs

        # --- Masked BCE: average only over annotated channels' voxels ---
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        n_vox = 1
        for d in spatial:
            n_vox *= logits.shape[d]
        bce_loss = (bce * mask_b).sum() / (valid_classes * n_vox)

        # --- Masked Dice: per (sample,class), then average over annotated ones ---
        inter = (probs * targets).sum(dim=spatial)                 # [B,C]
        denom = probs.sum(dim=spatial) + targets.sum(dim=spatial)  # [B,C]
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth) # [B,C]
        dice_loss = 1.0 - (dice * mask).sum() / valid_classes      # <-- masked reduction (the fix)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    loss = MaskedPartialLoss()

    # Build a 2-class case where class 1 is UNannotated in sample 0.
    B, C, H, W = 2, 2, 16, 16
    logits = torch.randn(B, C, H, W)
    targets = (torch.rand(B, C, H, W) > 0.7).float()
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # sample0: only class0 annotated

    full = loss(logits, targets, mask).item()

    # Corrupting the UNannotated channel (s0,c1) must NOT change the loss.
    t2 = targets.clone(); t2[0, 1] = 1 - t2[0, 1]      # flip its labels
    l2 = logits.clone();  l2[0, 1] = torch.randn(H, W) # and its logits
    changed = loss(l2, t2, mask).item()

    # A fully-annotated mask should equal a plain (unmasked) combined loss.
    mask_all = torch.ones(B, C)
    masked_all = loss(logits, targets, mask_all).item()

    print(f"loss (partial mask)            : {full:.6f}")
    print(f"loss after corrupting masked ch: {changed:.6f}  (should be identical)")
    print(f"identical? {abs(full - changed) < 1e-9}")
    print(f"loss with all-annotated mask   : {masked_all:.6f}")
    assert abs(full - changed) < 1e-9, "masked channel leaked into the loss!"
    print("PASS: unannotated classes are correctly ignored.")
