"""
E6 training — the SAME plain 3D U-Net as E4, on the 430-patient dirty set,
single fold 0. Two arms selected by --loss:

  --loss dirty   : ordinary Dice+BCE (DiceBCELoss3D). The un-annotated gland is
                   left as background and IS penalised -> reproduces the
                   annotation-gap cost on all 430 patients (the control arm).
  --loss masked  : MaskedPartialLoss (pipeline/masked_loss.py). The per-CASE mask
                   ignores an un-annotated gland's channel entirely -> captures the
                   extra 222 patients' data benefit WITHOUT the label penalty
                   (the key arm; §21's number-backed prediction).

Everything else is held identical to E4: architecture, patch [48,224,192],
Adam 1e-4, AMP, batch 2, 250 epochs / 250 iters, ReduceLROnPlateau, best-val
checkpointing, seed 42, nnU-Net fold-0 split (or --splits-json).

Run on the pod:
  python3 preprocess_e6.py --out ./preprocessed            # once, CPU (or upload cache)
  python3 train_e6.py --loss masked --preproc ./preprocessed/train --save-dir ./ckpt_masked430
  python3 train_e6.py --loss dirty  --preproc ./preprocessed/train --save-dir ./ckpt_dirty430
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from unet3d import UNet3D
from loss3d import DiceBCELoss3D
from masked_loss import MaskedPartialLoss
from dataloader_e6 import (MaskedPatch3DDataset, MaskedValDataset,
                           nnunet_fold0_split, PATCH)


def run_epoch(model, loader, arm, crit, opt, device, scaler, train, use_amp):
    model.train() if train else model.eval()
    tot = 0.0; n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, tgt, mask in loader:
            img, tgt, mask = img.to(device), tgt.to(device), mask.to(device)
            if train:
                opt.zero_grad()

            def compute():
                out = model(img)
                if arm == "masked":
                    return crit(out, tgt, mask)
                else:  # dirty: ordinary loss ignores the mask
                    return crit(out, tgt)[0]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = compute()
                if train:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = compute()
                if train:
                    loss.backward(); opt.step()
            tot += loss.item(); n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=["dirty", "masked"])
    ap.add_argument("--preproc", required=True, help="preprocessed/train dir (430 cases)")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--iters-per-epoch", type=int, default=250)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits-json", default=None,
                    help="optional nnU-Net Dataset003 splits_final.json for fold 0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"arm={args.loss} device={device} amp={use_amp} seed={args.seed}", flush=True)

    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.preproc, "*.npz")))
    if args.splits_json and os.path.exists(args.splits_json):
        sp = json.load(open(args.splits_json))[0]
        tr, val = sp["train"], sp["val"]
        print("using splits_final.json fold 0")
    else:
        tr, val = nnunet_fold0_split(ids)
        print("using reproduced KFold(5,seed=12345) fold 0")
    print(f"cases={len(ids)} train={len(tr)} val={len(val)}", flush=True)

    train_ds = MaskedPatch3DDataset(args.preproc, tr, augment=True,
                                    samples_per_epoch=args.batch * args.iters_per_epoch,
                                    seed=args.seed)
    val_ds = MaskedValDataset(args.preproc, val,
                              samples_per_epoch=max(50, args.batch * 25), seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = UNet3D(1, 2).to(device)
    print(f"UNet3D params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    crit = MaskedPartialLoss() if args.loss == "masked" else DiceBCELoss3D()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode="min", patience=8, factor=0.5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best = float("inf"); hist = []; noimp = 0
    ckpt = os.path.join(args.save_dir, "best_model.pth")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_l = run_epoch(model, train_loader, args.loss, crit, opt, device, scaler, True, use_amp)
        va_l = run_epoch(model, val_loader, args.loss, crit, opt, device, scaler, False, use_amp)
        sched.step(va_l)
        hist.append({"epoch": ep, "train_loss": tr_l, "val_loss": va_l,
                     "elapsed_s": round(time.time() - t0, 1)})
        json.dump(hist, open(os.path.join(args.save_dir, "training_history.json"), "w"), indent=2)
        print(f"Epoch {ep:03d}/{args.epochs} train {tr_l:.4f} val {va_l:.4f}", flush=True)
        if va_l < best:
            best = va_l; noimp = 0
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_loss": va_l, "patch": PATCH, "arm": args.loss}, ckpt)
            print(f"  saved best (val {va_l:.4f})", flush=True)
        else:
            noimp += 1
            if noimp >= args.patience:
                print(f"early stop @ {ep}", flush=True); break
    print(f"done. arm={args.loss} best val {best:.4f} -> {ckpt} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
