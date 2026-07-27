import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hdf5_dataloader import ParotidDataset
from loss_function import CombinedLoss
from swin_unet import SwinUNet


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = dice_loss = bce_loss = 0.0

    for i, (images, masks) in enumerate(loader):
        if i % 500 == 0:
            print(f"  Batch {i}/{len(loader)}", flush=True)
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            predictions = model(images)
            loss, dice, bce = criterion(predictions, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        dice_loss  += dice.item()
        bce_loss   += bce.item()

    n = len(loader)
    return total_loss / n, dice_loss / n, bce_loss / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = dice_loss = bce_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            if masks.sum() == 0:
                continue
            images = images.to(device)
            masks  = masks.to(device)
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss, dice, bce = criterion(predictions, masks)
            total_loss += loss.item()
            dice_loss  += dice.item()
            bce_loss   += bce.item()
            n_batches  += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_batches, dice_loss / n_batches, bce_loss / n_batches


def main():
    dataset_path = '/home/<hpc-user>/project/data/dataset.h5'
    split_path   = '/home/<hpc-user>/project/dataset_split.json'
    save_dir     = '/home/<hpc-user>/project/checkpoints/swin_unet'
    os.makedirs(save_dir, exist_ok=True)

    BATCH_SIZE    = 8
    NUM_EPOCHS    = 50
    LEARNING_RATE = 1e-4
    PATIENCE      = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = ParotidDataset('train', dataset_path, split_path, augment=True)
    val_ds   = ParotidDataset('val',   dataset_path, split_path, augment=False)

    train_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = SwinUNet(in_channels=1, out_channels=2).to(device)
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scaler    = torch.amp.GradScaler('cuda')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Swin-UNet parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    start_epoch   = 1
    best_val_loss = float('inf')
    history       = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {checkpoint['epoch']} (val_loss: {best_val_loss:.4f})")

        history_path = os.path.join(save_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
    else:
        print("No checkpoint found. Starting from scratch.")

    epochs_no_improve = 0

    print(f"\nStarting training from epoch {start_epoch} to {NUM_EPOCHS}...\n")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_dice, train_bce = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler)

        val_loss, val_dice, val_bce = validate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss':   val_loss,
            'val_dice':   val_dice
        })

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f} BCE: {train_bce:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Dice: {val_dice:.4f} BCE: {val_bce:.4f})")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
                'val_loss':             val_loss,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best validation loss : {best_val_loss:.4f}")
    print(f"Best model saved to  : {checkpoint_path}")


if __name__ == '__main__':
    main()
