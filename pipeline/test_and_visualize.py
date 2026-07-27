"""
test_and_visualize.py
=====================
Run the trained 3D nnU-Net parotid model on a folder of CT scans and produce
HUMAN-VIEWABLE results: 2D axial slice overlays (always) + an optional 3D surface
render of the predicted glands. Works on the locked test set OR on any scans you
choose.

IMPORTANT: this is a 3D model. It predicts on whole volumes, not single slices.
You point it at a FOLDER of scans; it segments each and draws the parotids.

------------------------------------------------------------------------------
INPUT MODES (pick one)
  --images  DIR   DIR contains nnU-Net-style volumes named  <case>_0000.nii.gz
  --patients DIR  DIR contains per-slice .npz patient folders (ML_Dataset_Final
                  style); each is reconstructed into a 3D volume on the fly
                  (uses build_volumes.py in this same folder).

PREDICTIONS
  --pred-dir DIR  Use already-computed predictions (<case>.nii.gz). Skips the
                  model entirely -> instant, needs no torch/nnU-Net. Use this to
                  view the test set (you already have ~/Desktop/pred_nomirror).
  (no --pred-dir) Runs the model itself (needs `pip install nnunetv2`). Mirroring
                  is disabled at inference (== --disable_tta) to keep L/R clean.

GROUND TRUTH (optional)
  --gt-dir  DIR   GT labelmaps (<case>.nii.gz, 1=parotid_r 2=parotid_l). If given,
                  GT is drawn as dashed contours and Dice/HD95 are computed.
                  (In --patients mode, GT is auto-built from the annotations.)

OUTPUT (--out DIR)
  <case>_slices.png  2D axial montage: CT + predicted R (red) / L (blue), GT dashed
  <case>_3d.png      3D gland surface render          (only with --render3d)
  metrics.csv        per-case Dice / HD95             (only with --gt-dir)
  index.html         click-through gallery of every case

------------------------------------------------------------------------------
EXAMPLES

# 1) See the TEST-SET results right now (no model run, uses your saved preds):
python test_and_visualize.py \
  --images   ~/Desktop/Dataset002_Parotid/imagesTs \
  --pred-dir ~/Desktop/pred_nomirror \
  --gt-dir   ~/Desktop/Dataset002_Parotid/labelsTs \
  --out      ~/Desktop/parotid_viz_testset --render3d

# 2) Test on NEW scans you provide as volumes (runs the model; needs nnunetv2):
python test_and_visualize.py \
  --images ~/Desktop/some_new_volumes \
  --out    ~/Desktop/parotid_viz_new --render3d

# 3) Test on raw per-slice patients from your dataset (reconstruct -> predict):
python test_and_visualize.py \
  --patients ~/Desktop/"Projects/ct scan project"/ML_Dataset_Final_subset \
  --out      ~/Desktop/parotid_viz_patients --render3d
"""
import os
import sys
import glob
import csv
import argparse
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")  # save PNGs, no GUI needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.ndimage import binary_erosion, distance_transform_edt

R_LABEL, L_LABEL = 1, 2
R_COLOR, L_COLOR = "#ff3b30", "#0a84ff"   # red = right parotid, blue = left parotid

DEFAULT_MODEL = os.path.expanduser(
    "~/Desktop/nnunet_results_Dataset002_noMirror/Dataset002_Parotid/"
    "nnUNetTrainer_250epochs_noMirror__nnUNetPlans__3d_fullres"
)


# ----------------------------------------------------------------------------- IO
def load_vol(path):
    """Return (array[Y,X,Z], spacing(sy,sx,sz))."""
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    zooms = nii.header.get_zooms()[:3]
    return arr, tuple(float(z) for z in zooms)


def window_ct(sl, hu_offset, wl, ww):
    """Map one CT slice to [0,1] using a soft-tissue window."""
    hu = sl.astype(np.float32) - hu_offset
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def pick_slices(pred, gt, n):
    """Choose up to n axial slices that actually contain glands, evenly spread."""
    mask = pred > 0
    if gt is not None:
        mask = mask | (gt > 0)
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if len(zs) == 0:                       # nothing predicted -> show the middle
        Z = pred.shape[2]
        return list(range(max(0, Z // 2 - n // 2), min(Z, Z // 2 + n // 2)))
    if len(zs) <= n:
        return list(zs)
    return list(zs[np.linspace(0, len(zs) - 1, n).astype(int)])


# ------------------------------------------------------------------ metrics
def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def hd95(pred, true, spacing):
    if pred.sum() == 0 and true.sum() == 0:
        return 0.0
    if pred.sum() == 0 or true.sum() == 0:
        return float("nan")
    # crop to union bbox for speed
    u = pred | true
    idx = np.array(np.where(u))
    lo = idx.min(1); hi = idx.max(1) + 1
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    p, t = pred[sl], true[sl]
    ps = p ^ binary_erosion(p)
    ts = t ^ binary_erosion(t)
    d_pt = distance_transform_edt(~t, sampling=spacing)[ps]
    d_tp = distance_transform_edt(~p, sampling=spacing)[ts]
    if len(d_pt) == 0 or len(d_tp) == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


# ------------------------------------------------------------------ 2D render
def render_2d(img, pred, gt, spacing, out_png, title, n_slices, hu_offset, wl, ww):
    zsel = pick_slices(pred, gt, n_slices)
    ncol = min(4, len(zsel)) or 1
    nrow = int(np.ceil(len(zsel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for i, z in enumerate(zsel):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(window_ct(img[:, :, z], hu_offset, wl, ww), cmap="gray",
                  interpolation="nearest")
        # predicted glands, semi-transparent fill
        for lbl, col in ((R_LABEL, R_COLOR), (L_LABEL, L_COLOR)):
            m = pred[:, :, z] == lbl
            if m.any():
                ax.imshow(np.ma.masked_where(~m, m), cmap=_solid_cmap(col),
                          alpha=0.45, interpolation="nearest", vmin=0, vmax=1)
        # ground truth, dashed contour in the matching side colour
        if gt is not None:
            for lbl, col in ((R_LABEL, R_COLOR), (L_LABEL, L_COLOR)):
                g = (gt[:, :, z] == lbl).astype(float)
                if g.any():
                    ax.contour(g, levels=[0.5], colors=[col], linewidths=1.1,
                               linestyles="dashed")
        ax.set_title(f"slice z={z}", fontsize=8)
        ax.axis("off")

    handles = [Patch(facecolor=R_COLOR, alpha=0.5, label="Right parotid (pred)"),
               Patch(facecolor=L_COLOR, alpha=0.5, label="Left parotid (pred)")]
    if gt is not None:
        handles.append(Patch(facecolor="none", edgecolor="k", linestyle="--",
                             label="Ground truth (dashed)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _solid_cmap(hexcol):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("_s", [hexcol, hexcol])


# ------------------------------------------------------------------ 3D render
def render_3d(pred, spacing, out_png, title):
    """Surface render of both glands. Uses marching cubes if scikit-image is
    available; otherwise falls back to a downsampled voxel plot."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa
    # crop to the glands (+ margin) so they fill the frame instead of floating
    # in a head-sized empty box
    fg = pred > 0
    if fg.any():
        idx = np.array(np.where(fg))
        lo = np.maximum(idx.min(1) - 6, 0)
        hi = np.minimum(idx.max(1) + 7, pred.shape)
        pred = pred[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    drew = False
    try:
        from skimage.measure import marching_cubes
        for lbl, col in ((R_LABEL, R_COLOR), (L_LABEL, L_COLOR)):
            m = (pred == lbl)
            if m.sum() < 10:
                continue
            verts, faces, _, _ = marching_cubes(m.astype(np.float32), level=0.5,
                                                 spacing=spacing)
            mesh = Poly3DCollection(verts[faces], alpha=0.6)
            mesh.set_facecolor(col)
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
            drew = True
    except ImportError:
        # fallback: no skimage -> plot downsampled voxels (coarser but works)
        step = 2
        for lbl, col in ((R_LABEL, R_COLOR), (L_LABEL, L_COLOR)):
            m = (pred[::step, ::step, ::step] == lbl)
            if m.sum() < 5:
                continue
            ax.voxels(m, facecolors=col, alpha=0.5, edgecolor=None)
            drew = True
        spacing = (spacing[0] * step, spacing[1] * step, spacing[2] * step)

    if not drew:
        plt.close(fig)
        return False

    # equal physical aspect
    nz = np.array(pred.shape) * np.array(spacing)
    try:
        ax.set_box_aspect(nz)
    except Exception:
        pass
    ax.set_xlabel("Y (mm)"); ax.set_ylabel("X (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(elev=15, azim=-70)
    ax.set_title(title, fontsize=12)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------------ prediction
def predict_folder(images_dir, out_dir, model_dir, device, checkpoint):
    """Run the nnU-Net model on images_dir -> out_dir. Needs nnunetv2 + torch."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    os.makedirs(out_dir, exist_ok=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True,
        use_mirroring=False,                      # == --disable_tta (clean L/R)
        perform_everything_on_device=(device != "cpu"),
        device=torch.device(device), verbose=False, allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=(0,), checkpoint_name=checkpoint)
    predictor.predict_from_files(
        images_dir, out_dir, save_probabilities=False, overwrite=False,
        num_processes_preprocessing=1, num_processes_segmentation_export=1)
    return out_dir


# ------------------------------------------------------------------ patients -> volumes
def reconstruct_patients(patients_dir, tmp_img_dir, tmp_gt_dir):
    """ML_Dataset_Final-style per-slice patient folders -> 3D NIfTI volumes."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_volumes import reconstruct_patient, _affine
    os.makedirs(tmp_img_dir, exist_ok=True)
    os.makedirs(tmp_gt_dir, exist_ok=True)
    made = []
    for entry in sorted(os.listdir(patients_dir)):
        pdir = os.path.join(patients_dir, entry)
        if not os.path.isdir(pdir):
            continue
        r = reconstruct_patient(pdir)
        if r is None:
            print(f"  skip {entry}: no usable slices")
            continue
        case = "".join(c if c.isalnum() else "_" for c in entry)
        aff = _affine(r["meta"]["z_spacing_mm"])
        nib.save(nib.Nifti1Image(r["image"], aff),
                 os.path.join(tmp_img_dir, f"{case}_0000.nii.gz"))
        # build a GT labelmap if this patient has parotid annotations
        lm = np.zeros(r["image"].shape, np.uint8)
        for organ, lid in (("PAROTID_R", R_LABEL), ("PAROTID_L", L_LABEL)):
            lm[r["labels"][organ].astype(bool)] = lid
        if lm.any():
            nib.save(nib.Nifti1Image(lm, aff), os.path.join(tmp_gt_dir, f"{case}.nii.gz"))
        made.append(case)
        print(f"  reconstructed {entry} -> {case}")
    return made


# ------------------------------------------------------------------ gallery
def write_gallery(out_dir, cases, has3d):
    rows = []
    for c, mean_d in cases:
        d = f" &nbsp; <b>Dice {mean_d:.3f}</b>" if mean_d is not None else ""
        imgs = f'<img src="{c}_slices.png" style="width:100%">'
        if has3d and os.path.exists(os.path.join(out_dir, f"{c}_3d.png")):
            imgs += f'<img src="{c}_3d.png" style="width:60%">'
        rows.append(f'<div style="margin:24px 0"><h3>{c}{d}</h3>{imgs}</div>')
    html = ("<html><head><meta charset='utf-8'><title>Parotid predictions</title>"
            "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:20px}"
            "img{border:1px solid #ddd;border-radius:6px;margin:4px 0}</style></head>"
            f"<body><h1>Parotid segmentation — {len(cases)} case(s)</h1>"
            + "".join(rows) + "</body></html>")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of <case>_0000.nii.gz volumes")
    src.add_argument("--patients", help="folder of per-slice npz patient dirs")
    ap.add_argument("--pred-dir", help="use existing predictions (skip the model)")
    ap.add_argument("--gt-dir", help="ground-truth labelmaps for overlay + metrics")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--render3d", action="store_true", help="also make 3D renders")
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--hu-offset", type=float, default=8192.0,
                    help="subtract to get HU (this dataset=8192; std DICOM=1024)")
    ap.add_argument("--wl", type=float, default=40.0, help="window level (HU)")
    ap.add_argument("--ww", type=float, default=400.0, help="window width (HU)")
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt_dir = args.gt_dir

    # 1. resolve input images (+ auto GT if reconstructing from patients)
    if args.patients:
        img_dir = os.path.join(args.out, "_recon_images")
        auto_gt = os.path.join(args.out, "_recon_gt")
        print("Reconstructing volumes from per-slice patients...")
        reconstruct_patients(args.patients, img_dir, auto_gt)
        if gt_dir is None and glob.glob(os.path.join(auto_gt, "*.nii.gz")):
            gt_dir = auto_gt
    else:
        img_dir = args.images

    image_files = sorted(glob.glob(os.path.join(img_dir, "*_0000.nii.gz")))
    if args.max_cases:
        image_files = image_files[:args.max_cases]
    if not image_files:
        sys.exit(f"No *_0000.nii.gz volumes found in {img_dir}")
    print(f"{len(image_files)} volume(s) to process.")

    # 2. resolve predictions
    if args.pred_dir:
        pred_dir = args.pred_dir
        print(f"Using existing predictions in {pred_dir}")
    else:
        pred_dir = os.path.join(args.out, "predictions")
        print(f"Running the model ({args.device}) -> {pred_dir}  (this can take a "
              f"few minutes per case on CPU)")
        predict_folder(img_dir, pred_dir, args.model_dir, args.device, args.checkpoint)

    # 3. per-case visualisation (+ metrics)
    metrics = []
    gallery = []
    for img_path in image_files:
        case = os.path.basename(img_path).replace("_0000.nii.gz", "")
        pred_path = os.path.join(pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"  ! no prediction for {case}, skipping")
            continue
        img, spacing = load_vol(img_path)
        pred, _ = load_vol(pred_path)
        pred = pred.astype(np.uint8)
        gt = None
        mean_d = None
        if gt_dir:
            gp = os.path.join(gt_dir, f"{case}.nii.gz")
            if os.path.exists(gp):
                gt = load_vol(gp)[0].astype(np.uint8)

        # metrics
        if gt is not None:
            dR = dice(pred == R_LABEL, gt == R_LABEL)
            dL = dice(pred == L_LABEL, gt == L_LABEL)
            hR = hd95(pred == R_LABEL, gt == R_LABEL, spacing)
            hL = hd95(pred == L_LABEL, gt == L_LABEL, spacing)
            mean_d = (dR + dL) / 2
            metrics.append([case, f"{dR:.4f}", f"{dL:.4f}", f"{mean_d:.4f}",
                            f"{hR:.2f}", f"{hL:.2f}"])
            title = f"{case}   Dice  R {dR:.3f} | L {dL:.3f}   HD95  R {hR:.1f} | L {hL:.1f} mm"
        else:
            title = f"{case}   (no ground truth)"

        render_2d(img, pred, gt, spacing, os.path.join(args.out, f"{case}_slices.png"),
                  title, args.n_slices, args.hu_offset, args.wl, args.ww)
        if args.render3d:
            render_3d(pred, spacing, os.path.join(args.out, f"{case}_3d.png"),
                      f"{case} — predicted glands (red=R, blue=L)")
        gallery.append((case, mean_d))
        print(f"  {case}: done" + (f"  (Dice {mean_d:.3f})" if mean_d is not None else ""))

    # 4. metrics.csv + gallery
    if metrics:
        with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "dice_R", "dice_L", "dice_mean", "hd95_R_mm", "hd95_L_mm"])
            w.writerows(metrics)
        arr = np.array([[float(x) for x in r[1:]] for r in metrics])
        print("\n==== SUMMARY (mean over cases) ====")
        print(f"Dice R {arr[:,0].mean():.4f} | L {arr[:,1].mean():.4f} | "
              f"mean {arr[:,2].mean():.4f}")
        print(f"HD95 R {np.nanmean(arr[:,3]):.2f} | L {np.nanmean(arr[:,4]):.2f} mm")

    write_gallery(args.out, gallery, args.render3d)
    print(f"\nOpen {os.path.join(args.out, 'index.html')} to flip through all cases.")


if __name__ == "__main__":
    main()
