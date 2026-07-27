"""
galcommon.py — shared config + rendering helpers for the E8 prediction gallery.
PURE RENDERING from existing prediction files. No training, no inference, no GPU.

Convention (kept everywhere): R = red #ff3b30, L = blue #0a84ff, GT = dashed contour
in the matching colour. Soft-tissue window (hu_offset 8192, WL 40, WW 400).

Labels in the nnU-Net-style .nii.gz are 1 = parotid_r, 2 = parotid_l — EXCEPT the
single-side E7 experts (left_both / right_both) which store their one gland as
label 1; each model carries a `remap` {pred_label: side} to normalise that.
"""
import os
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "ablation_study", "prediction_gallery")

R_COLOR, L_COLOR = "#ff3b30", "#0a84ff"
SIDE_COLOR = {"R": R_COLOR, "L": L_COLOR}
SIDE_NAME = {"R": "Right parotid", "L": "Left parotid"}
HU_OFFSET, WL, WW = 8192.0, 40.0, 400.0

CT_DIR = os.path.join(ROOT, "Parotid-Project/Datasets/Dataset002_Parotid/imagesTs")
GT_DIR = os.path.join(ROOT, "Parotid-Project/Datasets/Dataset002_Parotid/labelsTs")


def P(*parts):
    return os.path.join(ROOT, *parts)


# ------------------------------------------------------------------ model registry
# order = worst->best is set in ORDER below; each: key -> (label, dir, agg_dice, remap, note)
# remap maps a prediction label value -> anatomical side ('R'/'L').
DEFAULT_REMAP = {1: "R", 2: "L"}

MODELS = {
    "nnunet_1fold": dict(
        label="nnU-Net 3d (1 fold)", dir="Parotid-Project/Results/pred_nomirror",
        dice=0.8187, remap=DEFAULT_REMAP, note="reference"),
    "nnunet_ensemble": dict(
        label="nnU-Net 5-fold ensemble", dir="Parotid-Project/Results/pred_ensemble",
        dice=0.8202, remap=DEFAULT_REMAP, note="reference, BEST overall"),
    "E1_2d": dict(
        label="E1 nnU-Net 2d", dir="ablation_study/_pod_results/preds/E1_2d",
        dice=0.8117, remap=DEFAULT_REMAP, note="2D vs 3D: 3D worth only +0.007"),
    "E2_dirty": dict(
        label="E2 nnU-Net dirty-430", dir="ablation_study/_pod_results/preds/E2_dirty",
        dice=0.7726, remap=DEFAULT_REMAP, note="annotation gap, realistic net -0.046"),
    "E2b_gapped": dict(
        label="E2b nnU-Net gapped-208", dir="ablation_study/_pod_results/preds/E2b_gapped",
        dice=0.6899, remap=DEFAULT_REMAP, note="pure annotation-gap cost -0.129"),
    "E4_cc": dict(
        label="E4 hand-built 3D U-Net +CC",
        dir="ablation_study/_pod_results_B/E4_pred_test_cc",
        dice=0.7750, remap=DEFAULT_REMAP, note="largest-CC postproc"),
    "E6_masked_cc": dict(
        label="E6 custom U-Net masked-430 +CC",
        dir="ablation_study/_pod_results_B/E6_masked_loss/pred_masked430_cc",
        dice=0.7589, remap=DEFAULT_REMAP, note="annotation-gap FIX (masked loss) +CC"),
    "E6_dirty_cc": dict(
        label="E6 custom U-Net dirty-430 +CC",
        dir="ablation_study/_pod_results_B/E6_masked_loss/pred_dirty430_cc",
        dice=0.7296, remap=DEFAULT_REMAP, note="penalise arm, diverged +CC"),
    "E7_experts": dict(
        label="E7 per-side experts (combined)",
        dir="ablation_study/E7_experts/_pod_results/preds/experts_both",
        dice=0.8099, remap=DEFAULT_REMAP, note="decompose arm; L320+R318 experts"),
    "E7_left": dict(
        label="E7 left-only expert",
        dir="ablation_study/E7_experts/_pod_results/preds/left_both",
        dice=None, remap={1: "L"}, note="left specialist (label1=L)"),
    "E7_right": dict(
        label="E7 right-only expert",
        dir="ablation_study/E7_experts/_pod_results/preds/right_both",
        dice=None, remap={1: "R"}, note="right specialist (label1=R)"),
}

# display order worst -> best (by aggregate 3D Dice), single-side experts last
ORDER = ["E2b_gapped", "E6_dirty_cc", "E6_masked_cc", "E2_dirty", "E4_cc",
         "E7_experts", "E1_2d", "nnunet_1fold", "nnunet_ensemble",
         "E7_right", "E7_left"]

# comparison-grid subset (the clean apples-to-apples set, one panel each)
COMPARE_ORDER = ["nnunet_ensemble", "nnunet_1fold", "E1_2d", "E7_experts",
                 "E4_cc", "E2_dirty", "E6_masked_cc", "E6_dirty_cc", "E2b_gapped"]


# ------------------------------------------------------------------ IO
def load_vol(path):
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    zooms = nii.header.get_zooms()[:3]
    return arr, tuple(float(z) for z in zooms)


def norm_pred(pred, remap):
    """Return a dict side -> boolean 3D mask using the model's remap."""
    return {side: (pred == lbl) for lbl, side in remap.items()}


def load_gt(case):
    p = os.path.join(GT_DIR, f"{case}.nii.gz")
    if not os.path.exists(p):
        return None
    g = load_vol(p)[0].astype(np.uint8)
    return {"R": g == 1, "L": g == 2}


def window_ct(sl):
    hu = sl.astype(np.float32) - HU_OFFSET
    lo, hi = WL - WW / 2.0, WL + WW / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def _solid_cmap(hexcol):
    return LinearSegmentedColormap.from_list("_s", [hexcol, hexcol])


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


# ------------------------------------------------------------------ slice picking
def _union(masks):
    u = None
    for m in masks or []:
        if m is None:
            continue
        u = m if u is None else (u | m)
    return u


def pick_slices(masks, n, anchor=None):
    """Choose n gland-bearing z evenly spread. If `anchor` masks are given (the GT),
    the spread is taken over the anchor's z-extent so every panel is centred on a real
    contour — this avoids one-sided gland-tip slices (pred present / GT absent, or vice
    versa) that look like gross misses despite a high Dice. Falls back to the union."""
    a = _union(anchor)
    ref = a if (a is not None and a.any()) else _union(masks)
    if ref is None or not ref.any():
        Z = masks[0].shape[2]
        return list(range(max(0, Z // 2 - n // 2), min(Z, Z // 2 + n // 2)))
    zs = np.where(ref.any(axis=(0, 1)))[0]
    if len(zs) <= n:
        return list(zs)
    return list(zs[np.linspace(0, len(zs) - 1, n).astype(int)])


def head_bbox(ct, zc, margin=12):
    """Square in-plane crop around the HEAD (largest tissue component over the
    chosen slices; excludes the couch/table). ct is (Y,X,Z). Returns y0,y1,x0,x1."""
    thr = 7800 if np.median(ct) > 4000 else -300
    tissue2d = (ct[:, :, zc] > thr).any(axis=2)
    lab, nlab = ndimage.label(tissue2d)
    if nlab > 0:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, nlab + 1))
        tissue2d = (lab == (1 + int(np.argmax(sizes))))
    ys = np.where(tissue2d.any(axis=1))[0]
    xs = np.where(tissue2d.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return 0, ct.shape[0], 0, ct.shape[1]
    y0, y1, x0, x1 = ys[0], ys[-1], xs[0], xs[-1]
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(y1 - y0, x1 - x0) / 2 + margin
    y0 = max(0, int(cy - half)); y1 = min(ct.shape[0], int(cy + half))
    x0 = max(0, int(cx - half)); x1 = min(ct.shape[1], int(cx + half))
    return y0, y1, x0, x1


# ------------------------------------------------------------------ 2D montage
GT_DASH = "#f4efe3"   # cream: GT dashed contour stands out on black over any fill


def _overlay(ax, ct2d, pred2d, gt2d, z):
    """Draw one panel: CT (bilinear) + pred fill/contour + GT cream dashed."""
    ax.imshow(ct2d, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.autoscale(False)
    for side, m in pred2d.items():
        if m.any():
            rgba = np.zeros((*m.shape, 4))
            c = matplotlib.colors.to_rgb(SIDE_COLOR[side])
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = c
            rgba[..., 3] = m.astype(float) * 0.5
            ax.imshow(rgba, interpolation="nearest")
            for cl in measure.find_contours(m.astype(float), 0.5):
                ax.plot(cl[:, 1], cl[:, 0], color=SIDE_COLOR[side], lw=0.9)
    if gt2d:
        for side, gsl in gt2d.items():
            if gsl.any():
                for cl in measure.find_contours(gsl.astype(float), 0.5):
                    ax.plot(cl[:, 1], cl[:, 0], color=GT_DASH, lw=1.1,
                            ls=(0, (4, 2)), alpha=0.95)
    if z is not None:
        ax.text(0.03, 0.95, f"z={z}", transform=ax.transAxes, color="#cfcfcf",
                fontsize=8, va="top", ha="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_visible(False)


def render_montage(img, pred_sides, gt_sides, out_png, title, n_slices=12, dpi=200):
    """pred_sides / gt_sides: dict side->bool 3D mask (gt may be None). Dark theme,
    head-cropped so the head fills the frame (couch/table excluded)."""
    all_masks = list(pred_sides.values())
    if gt_sides:
        all_masks += list(gt_sides.values())
    # anchor the slice spread on GT so every panel is centred on a real contour
    zsel = pick_slices(all_masks, n_slices,
                       anchor=list(gt_sides.values()) if gt_sides else None)
    y0, y1, x0, x1 = head_bbox(img, zsel)
    ctw = window_ct(img)
    ncol = min(4, len(zsel)) or 1
    nrow = int(np.ceil(len(zsel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("black")
    for ax in axes.ravel():
        ax.axis("off"); ax.set_facecolor("black")
    for i, z in enumerate(zsel):
        ax = axes[i // ncol][i % ncol]
        pred2d = {s: m[y0:y1, x0:x1, z] for s, m in pred_sides.items()}
        gt2d = {s: g[y0:y1, x0:x1, z] for s, g in gt_sides.items()} if gt_sides else None
        _overlay(ax, ctw[y0:y1, x0:x1, z], pred2d, gt2d, z)
    handles = []
    for side in ("R", "L"):
        if side in pred_sides:
            handles.append(Patch(facecolor=SIDE_COLOR[side], alpha=0.6,
                                 label=f"{SIDE_NAME[side]} (pred)"))
    if gt_sides:
        handles.append(Line2D([0], [0], color=GT_DASH, lw=1.4, ls=(0, (4, 2)),
                              label="Ground truth"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                     fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.004),
                     labelcolor="white")
    fig.suptitle(title, fontsize=12.5, y=0.997, color="white")
    fig.subplots_adjust(left=0.006, right=0.994, top=0.955, bottom=0.055,
                        wspace=0.02, hspace=0.02)
    fig.savefig(out_png, dpi=dpi, facecolor="black")
    plt.close(fig)


# ------------------------------------------------------------------ 3D render
def render_3d(pred_sides, spacing, out_png, title, dpi=200):
    """Smooth, shaded surface render. Gaussian-smooths the mask to kill the 3 mm
    z-staircase, then plot_trisurf with lighting. Two 3/4 views, clean (no axes)."""
    # marching-cubes vertices per side (in mm), oriented RL / AP / SI
    meshes = {}
    allv = []
    for side, m3 in pred_sides.items():
        if m3.sum() < 30:
            continue
        ms = gaussian_filter(m3.astype(np.float32), sigma=0.9)
        try:
            verts, faces, _, _ = measure.marching_cubes(ms, level=0.4, spacing=spacing,
                                                        step_size=1)
        except Exception:
            continue
        P = np.column_stack([verts[:, 1], verts[:, 0], verts[:, 2]])  # RL, AP, SI
        meshes[side] = (P, faces)
        allv.append(P)
    if not meshes:
        return False
    V = np.concatenate(allv)
    rng = V.max(0) - V.min(0)

    fig = plt.figure(figsize=(7.6, 7.2))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("none")
    for side, (P, faces) in meshes.items():
        ax.plot_trisurf(P[:, 0], P[:, 1], faces, P[:, 2], color=SIDE_COLOR[side],
                        lw=0.0, antialiased=True, shade=True)
    ax.set_box_aspect(tuple(rng + 1e-6))
    ax.set_xlim(V[:, 0].min() - 3, V[:, 0].max() + 3)
    ax.set_ylim(V[:, 1].min() - 3, V[:, 1].max() + 3)
    ax.set_zlim(V[:, 2].min() - 3, V[:, 2].max() + 3)
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False); a.line.set_visible(False); a.set_ticks([])
    ax.set_axis_off()
    ax.view_init(elev=14, azim=-72)                 # 3/4 anterior: R & L side by side
    ax.set_position([0.0, 0.0, 1.0, 0.93])          # fill the frame
    fig.suptitle(title, fontsize=12, y=0.985)
    fig.savefig(out_png, dpi=dpi, transparent=False, facecolor="white")
    plt.close(fig)
    return True


def cases_in(model_dir):
    d = P(model_dir)
    return sorted(f.replace(".nii.gz", "") for f in os.listdir(d)
                  if f.endswith(".nii.gz") and f.startswith("PAR"))


def test_cases():
    return sorted(f.replace(".nii.gz", "") for f in os.listdir(GT_DIR)
                  if f.endswith(".nii.gz"))
