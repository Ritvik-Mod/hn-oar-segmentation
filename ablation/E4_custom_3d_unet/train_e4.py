"""
E4 training — plain 3D U-Net on nnU-Net-preprocessed Dataset002, single fold 0.
Adam 1e-4, AMP (CUDA), DiceBCE-3D loss, patch [48,224,192], batch 2 (plans.json).

Run on the pod:
  python3 preprocess_e4.py --out ./preprocessed         # once, CPU
  python3 train_e4.py --preproc ./preprocessed/train --save-dir ./ckpt_unet3d
"""
import os
import sys
import json
import time
import glob
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unet3d import UNet3D
from loss3d import DiceBCELoss3D
from dataloader_e4 import Patch3DDataset, ValPatchDataset, nnunet_fold0_split, PATCH


def run_epoch(model, loader, crit, opt, device, scaler, train, use_amp):
    model.train() if train else model.eval()
    tot = dl = bl = 0.0; n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, tgt in loader:
            img, tgt = img.to(device), tgt.to(device)
            if train:
                opt.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(img)
                    loss, d, b = crit(out, tgt)
                if train:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                out = model(img)
                loss, d, b = crit(out, tgt)
                if train:
                    loss.backward(); opt.step()
            tot += loss.item(); dl += d.item(); bl += b.item(); n += 1
    n = max(n, 1)
    return tot / n, dl / n, bl / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", required=True, help="preprocessed/train dir")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters-per-epoch", type=int, default=250)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits-json", default=None,
                    help="optional nnU-Net splits_final.json to use fold 0 exactly")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device={device} amp={use_amp} seed={args.seed}", flush=True)

    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.preproc, "*.npz")))
    if args.splits_json and os.path.exists(args.splits_json):
        sp = json.load(open(args.splits_json))[0]
        tr, val = sp["train"], sp["val"]
        print("using splits_final.json fold 0")
    else:
        tr, val = nnunet_fold0_split(ids)
        print("using reproduced KFold(5,seed=12345) fold 0")
    print(f"cases={len(ids)} train={len(tr)} val={len(val)}", flush=True)

    train_ds = Patch3DDataset(args.preproc, tr, augment=True,
                              samples_per_epoch=args.batch * args.iters_per_epoch, seed=args.seed)
    val_ds = ValPatchDataset(args.preproc, val,
                             samples_per_epoch=max(50, args.batch * 25), seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = UNet3D(1, 2).to(device)
    print(f"UNet3D params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    crit = DiceBCELoss3D()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode="min", patience=8, factor=0.5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best = float("inf"); hist = []; noimp = 0
    ckpt = os.path.join(args.save_dir, "best_model.pth")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_l = run_epoch(model, train_loader, crit, opt, device, scaler, True, use_amp)
        va_l = run_epoch(model, val_loader, crit, opt, device, scaler, False, use_amp)
        sched.step(va_l[0])
        hist.append({"epoch": ep, "train_loss": tr_l[0], "train_dice_loss": tr_l[1],
                     "val_loss": va_l[0], "val_dice_loss": va_l[1],
                     "elapsed_s": round(time.time() - t0, 1)})
        json.dump(hist, open(os.path.join(args.save_dir, "training_history.json"), "w"), indent=2)
        print(f"Epoch {ep:03d}/{args.epochs} train {tr_l[0]:.4f} val {va_l[0]:.4f} "
              f"(valDiceLoss {va_l[1]:.4f})", flush=True)
        if va_l[0] < best:
            best = va_l[0]; noimp = 0
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_loss": va_l[0], "patch": PATCH}, ckpt)
            print(f"  saved best (val {va_l[0]:.4f})", flush=True)
        else:
            noimp += 1
            if noimp >= args.patience:
                print(f"early stop @ {ep}", flush=True); break
    print(f"done. best val {best:.4f} -> {ckpt} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
