"""
A1 montage figures — qualitative evidence for the two Swin failure modes.

Renders, on the SAME axial slice, the CT with:
  - U-Net prediction (crisp, single-component) vs
  - Swin prediction (coarse 128x128-upsampled boundary + stray islands),
with the ground-truth contour overlaid on both.

Two figures:
  figures/montage_islands.png  — a slice chosen to show Swin stray islands far
      from the gland (H2), auto-selected as the slice with the most Swin
      predicted voxels that lie outside a dilation of the GT.
  figures/montage_coarse.png   — a zoom on the gland boundary showing Swin's
      blocky/stair-stepped edge (H1, the 128->512 bilinear upsample) vs U-Net's
      pixel-resolution edge.

Reuses P0's saved predictions; reads CT from Dataset002 imagesTs.
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label, generate_binary_structure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PREDS = os.path.join(ROOT, "ablation_study", "P0_phase1_on_test", "preds")
IMAGES_TS = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid", "imagesTs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
CONN = generate_binary_structure(3, 3)


def window(ct):
    hu = ct.astype(np.float32) - 8192.0
    return np.clip((np.clip(hu, -150, 250) + 150) / 400.0, 0, 1)


def load_preds(model, par):
    d = np.load(os.path.join(PREDS, f"{model}__{par}.npz"))
    # combine R (1) + L (2) into a labelmap for display
    lm = np.zeros_like(d["pred_r"], dtype=np.uint8)
    lm[d["pred_r"] > 0] = 1
    lm[d["pred_l"] > 0] = 2
    gt = np.zeros_like(lm)
    gt[d["gt_r"] > 0] = 1
    gt[d["gt_l"] > 0] = 2
    return lm, gt


def pick_island_case_slice(cases):
    """Choose (case, z) maximising Swin voxels lying outside a GT dilation."""
    best = (None, None, -1)
    for par in cases:
        sw, gt = load_preds("Swin_UNet", par)
        gtb = gt > 0
        far = sw.astype(bool) & (~binary_dilation(gtb, iterations=6))
        per_slice = far.reshape(far.shape[0], -1).sum(1)
        z = int(np.argmax(per_slice))
        if per_slice[z] > best[2] and (gt[z] > 0).sum() > 30:
            best = (par, z, int(per_slice[z]))
    return best[0], best[1]


def overlay(ax, ct2d, lm2d, gt2d, title):
    ax.imshow(ct2d, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*lm2d.shape, 4))
    rgba[lm2d == 1] = [1, 0.2, 0.2, 0.55]   # R red
    rgba[lm2d == 2] = [0.2, 0.4, 1, 0.55]   # L blue
    ax.imshow(rgba)
    # GT contour (dashed)
    for v, c in [(1, "yellow"), (2, "cyan")]:
        if (gt2d == v).any():
            ax.contour(gt2d == v, levels=[0.5], colors=c, linewidths=0.8, linestyles="--")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def crop_bbox(gt2d, pad=40):
    ys, xs = np.where(gt2d > 0)
    if len(ys) == 0:
        return slice(0, 512), slice(0, 512)
    y0, y1 = max(0, ys.min() - pad), min(512, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(512, xs.max() + pad)
    return slice(y0, y1), slice(x0, x1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--slice", type=int, default=None)
    args = ap.parse_args()

    cases = sorted({f.split("__")[1][:-4] for f in os.listdir(PREDS) if f.endswith(".npz")})
    if not cases:
        sys.exit("no predictions found; run P0 with --save-preds first")

    if args.case:
        par, z = args.case, args.slice
    else:
        par, z = pick_island_case_slice(cases)
    print(f"islands montage: case={par} slice={z}")

    ct = np.asarray(nib.load(os.path.join(IMAGES_TS, f"{par}_0000.nii.gz")).dataobj)
    ct = np.transpose(ct, (2, 0, 1))
    ct2d = window(ct[z])
    sw_lm, gt = load_preds("Swin_UNet", par)
    un_lm, _ = load_preds("U-Net", par)

    # count components on this slice for the caption
    def ncomp(lm2d):
        return sum(label(lm2d == v, structure=generate_binary_structure(2, 2))[1] for v in (1, 2))

    # ---- Figure 1: islands (full slice) ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    overlay(axes[0], ct2d, np.zeros_like(gt[z]), gt[z], f"{par} z={z}\nCT + GT contour")
    overlay(axes[1], ct2d, un_lm[z], gt[z], f"U-Net  (2D comps={ncomp(un_lm[z])})")
    overlay(axes[2], ct2d, sw_lm[z], gt[z], f"Swin-UNet  (2D comps={ncomp(sw_lm[z])})")
    fig.suptitle("Swin stray false-positive islands vs U-Net (dashed = ground truth)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "montage_islands.png"), dpi=140)
    plt.close(fig)

    # ---- Figure 2: coarse boundary zoom ----
    ys, xs = crop_bbox(gt[z], pad=25)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, lm, name in [(axes[0], None, "CT (zoom)"),
                         (axes[1], un_lm, "U-Net edge"),
                         (axes[2], sw_lm, "Swin edge (128->512 upsample)")]:
        ax.imshow(ct2d[ys, xs], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if lm is not None:
            sub = lm[z][ys, xs]
            rgba = np.zeros((*sub.shape, 4))
            rgba[sub == 1] = [1, 0.2, 0.2, 0.5]
            rgba[sub == 2] = [0.2, 0.4, 1, 0.5]
            ax.imshow(rgba, interpolation="nearest")
        gsub = gt[z][ys, xs]
        for v, c in [(1, "yellow"), (2, "cyan")]:
            if (gsub == v).any():
                ax.contour(gsub == v, levels=[0.5], colors=c, linewidths=1.0, linestyles="--")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Boundary detail, {par} z={z}: Swin's coarse 128x128 upsampled edge "
                 f"vs U-Net's pixel-resolution edge", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "montage_coarse.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {FIG}/montage_islands.png and montage_coarse.png")


if __name__ == "__main__":
    main()
