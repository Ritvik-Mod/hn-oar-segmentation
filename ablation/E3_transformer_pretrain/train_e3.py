"""
E3 training — retrain TransUNet / Swin-UNet in their ImageNet-pretrained
configuration, holding the Phase-1 2D protocol identical (CombinedLoss Dice+BCE,
Adam 1e-4, batch 8, AMP, WeightedRandomSampler 20.0 slice-level, hflip aug).

This is a COPY placed under ablation_study/E3_.../ (the frozen originals in
`dicom test/<model>_codes_from_hpc/*_train.py` are untouched). Only two things
change vs the frozen scripts: (1) the model is built by models_pretrained, and
(2) paths are CLI args instead of hardcoded HPC absolutes.

Run on the pod (CUDA). Example:
  python3 train_e3.py --model transunet_pretrained \
      --data $DATA/dataset.h5 --split $DATA/dataset_split.json \
      --save-dir ./ckpt_transunet_pretrained
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
# reuse frozen dataloader + loss (read-only)
sys.path.insert(0, os.path.join(ROOT, "dicom test", "unet_codes_from_hpc"))
from hdf5_dataloader import ParotidDataset
from loss_function import CombinedLoss
from models_pretrained import build_transunet_pretrained, build_swin_pretrained


def build(model_name, pretrained):
    if model_name == "transunet_pretrained":
        return build_transunet_pretrained(pretrained=pretrained)
    if model_name == "swin_pretrained":
        return build_swin_pretrained(pretrained=pretrained)
    raise ValueError(model_name)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total = dl = bl = 0.0
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                pred = model(images)
                loss, dice, bce = criterion(pred, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images)
            loss, dice, bce = criterion(pred, masks)
            loss.backward()
            optimizer.step()
        total += loss.item(); dl += dice.item(); bl += bce.item()
        if i % 50 == 0:
            print(f"  batch {i}/{len(loader)} loss {loss.item():.4f}", flush=True)
    n = len(loader)
    return total / n, dl / n, bl / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total = dl = bl = 0.0; nb = 0
    for images, masks in loader:
        if masks.sum() == 0:
            continue
        images, masks = images.to(device), masks.to(device)
        if use_amp:
            with torch.amp.autocast("cuda"):
                pred = model(images)
                loss, dice, bce = criterion(pred, masks)
        else:
            pred = model(images)
            loss, dice, bce = criterion(pred, masks)
        total += loss.item(); dl += dice.item(); bl += bce.item(); nb += 1
    if nb == 0:
        return 0.0, 0.0, 0.0
    return total / nb, dl / nb, bl / nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["transunet_pretrained", "swin_pretrained"])
    ap.add_argument("--data", required=True, help="path to dataset.h5")
    ap.add_argument("--split", required=True, help="path to dataset_split.json")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples-per-epoch", type=int, default=0,
                    help="cap samples drawn per epoch (0 = full pass ~ Phase-1). "
                         "Budget knob: e.g. 12000 = 1500 iters/epoch at batch 8.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="ablate: train the same arch from scratch (sanity/control)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device={device} amp={use_amp} model={args.model} "
          f"pretrained={not args.no_pretrained} seed={args.seed}", flush=True)

    train_ds = ParotidDataset("train", args.data, args.split, augment=True)
    val_ds = ParotidDataset("val", args.data, args.split, augment=False)
    weights = train_ds.get_sample_weights()
    n_samples = args.samples_per_epoch if args.samples_per_epoch > 0 else len(weights)
    print(f"samples/epoch: {n_samples} ({'full pass' if args.samples_per_epoch==0 else 'capped'}) "
          f"of {len(weights)} train slices", flush=True)
    sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model, info = build(args.model, pretrained=not args.no_pretrained)
    model = model.to(device)
    print("build info:", info, flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M", flush=True)

    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best = float("inf"); history = []; no_improve = 0
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        va = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step(va[0])
        history.append({"epoch": epoch, "train_loss": tr[0], "train_dice": tr[1],
                        "val_loss": va[0], "val_dice": va[1],
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"Epoch {epoch:03d}/{args.epochs} | train {tr[0]:.4f} | val {va[0]:.4f}", flush=True)
        json.dump(history, open(os.path.join(args.save_dir, "training_history.json"), "w"), indent=2)
        if va[0] < best:
            best = va[0]; no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": va[0], "model_name": args.model,
                        "pretrained": not args.no_pretrained}, ckpt_path)
            print(f"  saved best (val {va[0]:.4f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop @ epoch {epoch}", flush=True)
                break
    print(f"done. best val {best:.4f} -> {ckpt_path} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
