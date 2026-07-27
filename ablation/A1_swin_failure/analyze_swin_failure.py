"""
A1 — Why Swin-UNet's boundary metrics (HD95, Surface-Dice) are disproportionately
bad relative to its Dice.  ANALYSIS ONLY — no training, no GPU.

Reuses the prediction volumes saved by P0
(`ablation_study/P0_phase1_on_test/preds/<model>__<PARxxxx>.npz`, keys pred_r,
pred_l, gt_r, gt_l), so no re-inference is needed.

Two hypotheses from the master doc (11.4, 13) tested on the locked 43-case test set:
  H1 (coarse output): Swin's final logits are 128x128 bilinearly upsampled to 512x512
     (swin_unet.py:402-404) -> inherently coarse boundaries -> inflated HD95.
  H2 (underfit / stray islands): Swin floored at val_loss ~0.21 vs ~0.16 CNNs ->
     scattered false-positive voxels far from the gland -> HD95 (95th pctile) blows up.

Metrics per predicted gland (Swin vs U-Net, 43 cases x {R,L} = 86 glands each):
  (a) connected-component count + stray-island burden;
  (b) distance (mm) of the farthest stray island from the main gland (true spacing);
  (c) HD95 on the full prediction vs on the largest-component-only prediction — if
      cleaning islands collapses Swin's HD95 but not U-Net's, H2 is the driver.

Speed: all distance transforms run on a bbox crop around the masks (like
eval_testset.py), so the whole thing is a couple of minutes. Prints per-case progress.
"""
import os
import sys
import json
import csv
import time
import numpy as np
from scipy.ndimage import (label, distance_transform_edt, binary_erosion,
                           generate_binary_structure)
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PREDS = os.path.join(ROOT, "ablation_study", "P0_phase1_on_test", "preds")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SPACING = (3.0, 0.977, 0.977)           # (z, y, x) mm, true anisotropic
CONN = generate_binary_structure(3, 3)  # 26-connectivity in 3D


def _bbox(*masks, margin=2):
    """Combined nonzero bbox (with margin) over the given boolean volumes."""
    any_ = np.zeros(masks[0].shape, bool)
    for m in masks:
        any_ |= m.astype(bool)
    if not any_.any():
        return None
    idx = np.where(any_)
    sl = []
    for d in range(3):
        lo = max(0, int(idx[d].min()) - margin)
        hi = min(any_.shape[d], int(idx[d].max()) + 1 + margin)
        sl.append(slice(lo, hi))
    return tuple(sl)


def hd95(pred, true, spacing=SPACING):
    if pred.sum() == 0 and true.sum() == 0:
        return 0.0
    if pred.sum() == 0 or true.sum() == 0:
        return np.nan
    bb = _bbox(pred, true, margin=2)
    pb, tb = pred[bb].astype(bool), true[bb].astype(bool)
    ps = pb ^ binary_erosion(pb)
    ts = tb ^ binary_erosion(tb)
    dpt = distance_transform_edt(~tb, sampling=spacing)
    dtp = distance_transform_edt(~pb, sampling=spacing)
    d = np.concatenate([dpt[ps], dtp[ts]])
    if d.size == 0:
        return np.nan
    return float(np.percentile(d, 95))


def dice(pred, true):
    s = pred.sum() + true.sum()
    return 1.0 if s == 0 else float(2.0 * np.sum(pred * true) / s)


def largest_cc(mask):
    lab, n = label(mask, structure=CONN)
    if n <= 1:
        return mask.astype(np.uint8)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (lab == int(np.argmax(sizes))).astype(np.uint8)


def stray_analysis(pred):
    lab, n = label(pred, structure=CONN)
    total = int(pred.sum())
    if n == 0:
        return dict(n_components=0, main_vox=0, stray_components=0, stray_vox=0,
                    stray_frac=0.0, farthest_stray_mm=0.0)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    main_id = int(np.argmax(sizes))
    main_mask = (lab == main_id)
    main_vox = int(sizes[main_id])
    stray_vox = total - main_vox
    stray_comps = n - 1
    farthest = 0.0
    if stray_comps > 0:
        # distance of stray voxels to the main component, true spacing, on a bbox crop
        bb = _bbox(pred, margin=1)
        dist_to_main = distance_transform_edt(~main_mask[bb], sampling=SPACING)
        stray_mask = pred[bb].astype(bool) & (~main_mask[bb])
        if stray_mask.any():
            farthest = float(dist_to_main[stray_mask].max())
    return dict(n_components=int(n), main_vox=main_vox, stray_components=int(stray_comps),
                stray_vox=int(stray_vox),
                stray_frac=float(stray_vox / total) if total else 0.0,
                farthest_stray_mm=farthest)


def load(model, par):
    return np.load(os.path.join(PREDS, f"{model}__{par}.npz"))


def main():
    if not os.path.isdir(PREDS) or not os.listdir(PREDS):
        sys.exit(f"No predictions in {PREDS} — run P0 with --save-preds first.")
    cases = sorted({f.split("__")[1][:-4] for f in os.listdir(PREDS) if f.endswith(".npz")})
    models = sorted({f.split("__")[0] for f in os.listdir(PREDS) if f.endswith(".npz")})
    print(f"cases={len(cases)} models={models}", flush=True)
    focus = [m for m in ["Swin_UNet", "U-Net"] if m in models]

    rows = []
    t0 = time.time()
    for i, par in enumerate(cases):
        for model in focus:
            d = load(model, par)
            for side in ("r", "l"):
                pred = d[f"pred_{side}"]; gt = d[f"gt_{side}"]
                sa = stray_analysis(pred)
                lcc = largest_cc(pred)
                rows.append(dict(model=model, case=par, side=side.upper(),
                                 dice=dice(pred, gt),
                                 hd95_full=hd95(pred, gt),
                                 hd95_lcc=hd95(lcc, gt),
                                 pred_vox=int(pred.sum()), **sa))
        print(f"  [{i+1}/{len(cases)}] {par}  ({time.time()-t0:.0f}s)", flush=True)

    fields = ["model", "case", "side", "dice", "n_components", "stray_components",
              "main_vox", "stray_vox", "stray_frac", "farthest_stray_mm",
              "hd95_full", "hd95_lcc", "pred_vox"]
    with open(os.path.join(HERE, "per_gland.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"wrote per_gland.csv ({len(rows)} glands)", flush=True)

    def agg(m):
        rs = [r for r in rows if r["model"] == m]
        ncc = np.array([r["n_components"] for r in rs])
        hf = np.array([r["hd95_full"] for r in rs], float)
        hl = np.array([r["hd95_lcc"] for r in rs], float)
        return dict(n_glands=len(rs), mean_dice=float(np.mean([r["dice"] for r in rs])),
                    mean_components=float(ncc.mean()), max_components=int(ncc.max()),
                    pct_multi_component=float(100*np.mean(ncc > 1)),
                    mean_stray_components=float(np.mean([r["stray_components"] for r in rs])),
                    mean_stray_frac_pct=float(100*np.mean([r["stray_frac"] for r in rs])),
                    mean_farthest_stray_mm=float(np.mean([r["farthest_stray_mm"] for r in rs])),
                    max_farthest_stray_mm=float(np.max([r["farthest_stray_mm"] for r in rs])),
                    hd95_full=float(np.nanmean(hf)), hd95_lcc=float(np.nanmean(hl)),
                    hd95_drop_from_lcc=float(np.nanmean(hf) - np.nanmean(hl)))
    summary = {m: agg(m) for m in focus}

    paired = {}
    if set(focus) == {"Swin_UNet", "U-Net"}:
        key = lambda r: (r["case"], r["side"])
        sw = {key(r): r for r in rows if r["model"] == "Swin_UNet"}
        un = {key(r): r for r in rows if r["model"] == "U-Net"}
        common = sorted(set(sw) & set(un))
        for metric in ["n_components", "stray_components", "stray_frac", "farthest_stray_mm"]:
            a = np.array([sw[k][metric] for k in common], float)
            b = np.array([un[k][metric] for k in common], float)
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = float("nan")
            paired[metric] = dict(swin_mean=float(a.mean()), unet_mean=float(b.mean()),
                                  swin_gt_unet=int((a > b).sum()), n=len(common),
                                  wilcoxon_p=float(p))

    out = dict(summary=summary, paired_swin_vs_unet=paired,
               note="true spacing (3.0,0.977,0.977); hd95_full vs hd95_lcc = "
                    "effect of removing stray islands")
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
