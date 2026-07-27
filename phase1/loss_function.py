import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        predictions: (B, 2, 512, 512) — raw logits from model
        targets:     (B, 2, 512, 512) — binary ground truth masks
        """
        predictions = torch.sigmoid(predictions)

        # Flatten spatial AND batch dimensions
        B, C, H, W = predictions.shape
        predictions = predictions.view(B, C, -1)  # (B, 2, 262144)
        targets     = targets.view(B, C, -1)       # (B, 2, 262144)

        # IMPORTANT: Sum across BOTH spatial (dim=2) AND batch (dim=0)
        # This calculates one global Dice score for the Right channel and one for the Left
        intersection = (predictions * targets).sum(dim=(0, 2))
        union        = predictions.sum(dim=(0, 2)) + targets.sum(dim=(0, 2))

        dice_per_channel = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average the loss between Right and Left channels
        return 1.0 - dice_per_channel.mean()


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        predictions: (B, 2, 512, 512) — raw logits
        targets:     (B, 2, 512, 512) — binary ground truth
        """
        return self.bce(predictions, targets)


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice_loss   = DiceLoss(smooth=smooth)
        self.bce_loss    = BCELoss()

    def forward(self, predictions, targets):
        """
        predictions: (B, 2, 512, 512) — raw logits from model
        targets:     (B, 2, 512, 512) — binary ground truth masks
        Returns: total loss, dice loss, bce loss (for logging)
        """
        dice = self.dice_loss(predictions, targets)
        bce  = self.bce_loss(predictions, targets)
        total = self.dice_weight * dice + self.bce_weight * bce
        return total, dice, bce


if __name__ == '__main__':
    # Sanity check
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)

    # Simulate a batch — random logits and binary masks
    B, C, H, W = 8, 2, 512, 512
    predictions = torch.randn(B, C, H, W,requires_grad=True)        # raw logits from model
    targets     = torch.zeros(B, C, H, W)

    # Put some fake parotid pixels in targets
    targets[:, 0, 100:150, 100:150] = 1.0        # PAROTID_R
    targets[:, 1, 200:240, 300:340] = 1.0        # PAROTID_L

    total_loss, dice_loss, bce_loss = criterion(predictions, targets)

    print(f"Total loss : {total_loss.item():.4f}")
    print(f"Dice loss  : {dice_loss.item():.4f}")
    print(f"BCE loss   : {bce_loss.item():.4f}")

    # Verify gradients flow
    total_loss.backward()
    print(f"Gradient check passed — loss is differentiable")

    # Test with perfect predictions
    perfect_preds = targets * 10 - 5  # high logits where mask=1, low where mask=0
    total_p, dice_p, bce_p = criterion(perfect_preds, targets)
    print(f"\nPerfect prediction loss: {total_p.item():.4f} (should be near 0)")

    # Test with all-zero predictions (worst case)
    zero_preds = torch.zeros(B, C, H, W)
    total_z, dice_z, bce_z = criterion(zero_preds, targets)
    print(f"All-zero prediction loss: {total_z.item():.4f} (should be > 0)")