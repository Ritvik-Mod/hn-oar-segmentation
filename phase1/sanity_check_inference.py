"""
Sanity check: load the trained Attention U-Net checkpoint and run inference on a
few real validation slices, saving prediction-vs-ground-truth overlays.

Purpose: confirm the checkpoint loads, the architecture matches, and predictions
are sane BEFORE building the demo / retraining on top of it.
"""
import os, glob, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/ritvikmod/Desktop/Projects/ct scan project"
sys.path.insert(0, os.path.join(ROOT, "dicom test", "attention_unet_codes_from_hpc"))
from attention_unet import AttentionUNet  # noqa: E402

DATA_DIR   = os.path.join(ROOT, "ML_Dataset_Final")
SPLIT_JSON = os.path.join(ROOT, "patient classification", "dataset_split.json")
CKPT       = os.path.join(ROOT, "checkpoints", "attention_unet", "best_model_attention.pth")
OUT_DIR    = os.path.join(ROOT, "sanity_check_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def window_and_normalise(image):
    image_hu = image.astype(np.float32) * 1.0 + (-8192.0)
    image_clipped = np.clip(image_hu, -150.0, 250.0)
    return (image_clipped - (-150.0)) / (250.0 - (-150.0))


def dice(pred, true):
    s = pred.sum() + true.sum()
    return 1.0 if s == 0 else float(2.0 * (pred * true).sum() / s)


def patient_dir(pid):
    d = os.path.join(DATA_DIR, pid)
    return d if os.path.isdir(d) else None


def slices_with_parotid(pdir):
    """Return list of (npz_path, has_R, has_L) for slices containing a parotid mask."""
    out = []
    for f in glob.glob(os.path.join(pdir, "**", "*.npz"), recursive=True):
        try:
            d = np.load(f)
        except Exception:
            continue
        hasR = "PAROTID_R" in d.files and d["PAROTID_R"].sum() > 0
        hasL = "PAROTID_L" in d.files and d["PAROTID_L"].sum() > 0
        if hasR or hasL:
            out.append((f, hasR, hasL))
    return sorted(out)


def main():
    device = torch.device("cpu")
    print(f"Loading checkpoint: {CKPT}")
    model = AttentionUNet(in_channels=1, out_channels=2).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  loaded OK | epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")

    val = json.load(open(SPLIT_JSON))["val"]
    print(f"Validation patients in split: {len(val)}")

    # find first val patient present locally that has parotid slices
    chosen, cand = None, None
    for pid in val:
        pdir = patient_dir(pid)
        if pdir is None:
            continue
        cand = pdir
        sl = slices_with_parotid(pdir)
        if len(sl) >= 4:
            chosen = (pid, pdir, sl)
            break
    if chosen is None:
        print("No local val patient with >=4 parotid slices found; falling back to any local patient.")
        if cand is None:
            print("ERROR: no val patients present locally.")
            return
        pid = os.path.basename(cand)
        chosen = (pid, cand, slices_with_parotid(cand))

    pid, pdir, sl = chosen
    print(f"\nChosen patient: {pid}  ({len(sl)} parotid slices)")

    # sample up to 6 evenly spaced slices
    idxs = np.linspace(0, len(sl) - 1, min(6, len(sl))).astype(int)
    dice_r_all, dice_l_all = [], []

    for n, i in enumerate(idxs):
        f, hasR, hasL = sl[i]
        d = np.load(f)
        img = d["image"]
        gt_r = d["PAROTID_R"] if "PAROTID_R" in d.files else np.zeros((512, 512), np.uint8)
        gt_l = d["PAROTID_L"] if "PAROTID_L" in d.files else np.zeros((512, 512), np.uint8)

        x = torch.tensor(window_and_normalise(img)).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).squeeze(0).numpy()
        pred_r = (prob[0] > 0.5).astype(np.uint8)
        pred_l = (prob[1] > 0.5).astype(np.uint8)

        dr, dl = dice(pred_r, gt_r), dice(pred_l, gt_l)
        if gt_r.sum() > 0: dice_r_all.append(dr)
        if gt_l.sum() > 0: dice_l_all.append(dl)

        disp = np.clip(img.astype(np.float32) - 8192.0, -150, 250)
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
        for a in ax:
            a.imshow(disp, cmap="gray"); a.axis("off")
        ax[0].contour(gt_r, colors="lime", linewidths=1.2)
        ax[0].contour(gt_l, colors="cyan", linewidths=1.2)
        ax[0].set_title("Ground truth (R=green, L=cyan)")
        ax[1].contour(pred_r, colors="red", linewidths=1.2)
        ax[1].contour(pred_l, colors="orange", linewidths=1.2)
        ax[1].set_title(f"Prediction  Dice R={dr:.2f}  L={dl:.2f}")
        fig.suptitle(f"{pid}  |  {os.path.basename(f)}")
        fig.tight_layout()
        outp = os.path.join(OUT_DIR, f"{n:02d}_{pid.replace('~','-')}_{os.path.basename(f).replace('.npz','')}.png")
        fig.savefig(outp, dpi=90); plt.close(fig)
        print(f"  slice {os.path.basename(f):22s}  Dice R={dr:.3f}  L={dl:.3f}  -> {os.path.basename(outp)}")

    print("\n=== SANITY CHECK SUMMARY ===")
    if dice_r_all:
        print(f"  PAROTID_R mean 2D Dice over {len(dice_r_all)} slices: {np.mean(dice_r_all):.3f}")
    if dice_l_all:
        print(f"  PAROTID_L mean 2D Dice over {len(dice_l_all)} slices: {np.mean(dice_l_all):.3f}")
    print(f"  Overlays saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
