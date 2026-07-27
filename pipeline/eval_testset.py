"""
Evaluate nnU-Net predictions on the locked test set, using the SAME metrics as
the original project so numbers are comparable to results_tracker.csv.

Compares predicted labelmaps (0=bg,1=parotid_r,2=parotid_l) against the held-out
labelsTs. Metrics are computed per structure (R, L) in 3D, then averaged.

Unlike the original 2D-stacked eval, HD95/Surface-Dice here use TRUE anisotropic
voxel spacing (0.977 x 0.977 x 3.0 mm), so boundary metrics are physically correct.
"""
import os, glob, json, argparse, csv
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion

SPACING = (0.977, 0.977, 3.0)  # (y, x, z) mm


def dice(pred, true):
    s = pred.sum() + true.sum()
    return 1.0 if s == 0 else float(2.0 * (pred * true).sum() / s)


def tversky(pred, true, alpha=0.3, beta=0.7):
    TP = np.sum(pred * true); FP = np.sum(pred * (1 - true)); FN = np.sum((1 - pred) * true)
    if true.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return TP / (TP + alpha * FP + beta * FN)


def _bbox_crop(a, b, margin=4):
    """Crop both volumes to the bounding box of their union (+margin) for speed.
    Safe for surface distances: all surface voxels of both masks lie inside."""
    u = (a > 0) | (b > 0)
    if not u.any():
        return a, b
    idx = np.array(np.where(u))
    lo = np.maximum(idx.min(1) - margin, 0)
    hi = np.minimum(idx.max(1) + margin + 1, a.shape)
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    return a[sl], b[sl]


def boundary_metrics(pred, true, tol_mm=3.0):
    """Return (hd95_mm, surface_dice) computing the distance transform once."""
    if pred.sum() == 0 and true.sum() == 0:
        return 0.0, 1.0
    if pred.sum() == 0 or true.sum() == 0:
        return np.nan, 0.0
    p, t = _bbox_crop(pred, true)
    pb, tb = p.astype(bool), t.astype(bool)
    ps = pb ^ binary_erosion(pb)
    ts = tb ^ binary_erosion(tb)
    d_pt = distance_transform_edt(~tb, sampling=SPACING)
    d_tp = distance_transform_edt(~pb, sampling=SPACING)
    sp, st = d_pt[ps], d_tp[ts]
    if len(sp) == 0 or len(st) == 0:
        return np.nan, 0.0
    hd = float(np.percentile(np.concatenate([sp, st]), 95))
    num = (sp <= tol_mm).sum() + (st <= tol_mm).sum()
    den = ps.sum() + ts.sum()
    sd = float(num / den) if den > 0 else 1.0
    return hd, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="nnU-Net predicted labelmaps (PARxxxx.nii.gz)")
    ap.add_argument("--gt-dir", required=True, help="labelsTs (ground-truth labelmaps)")
    ap.add_argument("--model-name", default="nnUNet_Parotid_3dfullres")
    ap.add_argument("--results-csv", default=None)
    args = ap.parse_args()

    per = {1: {"dice": [], "tv": [], "hd": [], "sd": []},
           2: {"dice": [], "tv": [], "hd": [], "sd": []}}
    n = 0
    for gt_path in sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz"))):
        case = os.path.basename(gt_path)
        pred_path = os.path.join(args.pred_dir, case)
        if not os.path.exists(pred_path):
            print(f"  ! no prediction for {case}, skipping"); continue
        gt = np.asarray(nib.load(gt_path).dataobj)
        pr = np.asarray(nib.load(pred_path).dataobj)
        n += 1
        for cls in (1, 2):
            g = (gt == cls).astype(np.uint8); p = (pr == cls).astype(np.uint8)
            if g.sum() == 0:  # no GT for this side in this case -> skip (don't inflate)
                continue
            hd_v, sd_v = boundary_metrics(p, g)
            per[cls]["dice"].append(dice(p, g))
            per[cls]["tv"].append(tversky(p, g))
            per[cls]["hd"].append(hd_v)
            per[cls]["sd"].append(sd_v)

    def agg(key):
        vals = per[1][key] + per[2][key]
        fn = np.nanmean if key == "hd" else np.mean
        return float(fn(vals)) if vals else float("nan")

    d, tv, hd_, sd = agg("dice"), agg("tv"), agg("hd"), agg("sd")
    print(f"\nEvaluated {n} test cases")
    print("=" * 46)
    print(f"{args.model_name}")
    print("=" * 46)
    print(f"3D Dice            : {d:.4f}")
    print(f"Clinical Tversky   : {tv:.4f}")
    print(f"HD95 (mm)          : {hd_:.2f}")
    print(f"Surface Dice (3mm) : {sd:.4f}")
    print(f"  R Dice {np.mean(per[1]['dice']):.4f} | L Dice {np.mean(per[2]['dice']):.4f}")

    if args.results_csv:
        new = not os.path.isfile(args.results_csv)
        with open(args.results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["Model_Name", "3D_Dice", "Tversky", "HD95_mm", "Surface_Dice_3mm", "Set"])
            w.writerow([args.model_name, f"{d:.4f}", f"{tv:.4f}", f"{hd_:.2f}", f"{sd:.4f}", "TEST"])
        print(f"Appended to {args.results_csv}")


if __name__ == "__main__":
    main()
