"""Render the P0 from-scratch 2D baselines (U-Net / Attention / TransUNet / Swin).
Predictions are .npz (pred_r/pred_l/gt_r/gt_l) in (Z,Y,X); verified identical
geometry to Dataset002 imagesTs, so we overlay on that CT (transposed to Y,X,Z)."""
import os, json, time
import numpy as np
import galcommon as g
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

P0_DIR = g.P("ablation_study/_pod_results_B/P0_preds")
P0 = {  # key -> (label, npz-prefix, aggregate test Dice from MASTER §20.0)
    "P0_unet":      ("P0 U-Net (from scratch, 2D)", "U-Net", 0.7390),
    "P0_attention": ("P0 Attention U-Net (2D)", "Attention_UNet", 0.7434),
    "P0_transunet": ("P0 TransUNet (2D)", "TransUNet", 0.7313),
    "P0_swin":      ("P0 Swin-UNet (2D)", "Swin_UNet", 0.7156),
}
P0_ORDER = ["P0_swin", "P0_transunet", "P0_unet", "P0_attention"]
cases = g.test_cases()


def load_p0(prefix, case):
    d = np.load(os.path.join(P0_DIR, f"{prefix}__{case}.npz"))
    tr = lambda a: np.transpose(a.astype(bool), (1, 2, 0))  # (Z,Y,X)->(Y,X,Z)
    ps = {"R": tr(d["pred_r"]), "L": tr(d["pred_l"])}
    gt = {"R": tr(d["gt_r"]), "L": tr(d["gt_l"])}
    return ps, gt


# ---- per-case montages + 3D + per-case Dice ----
manifest = {}
percase_p0 = {c: {} for c in cases}
t0 = time.time()
for key in P0_ORDER:
    label, prefix, agg = P0[key]
    odir = os.path.join(g.OUT, key)
    os.makedirs(odir, exist_ok=True)
    rows = []
    for c in cases:
        img = g.load_vol(os.path.join(g.CT_DIR, f"{c}_0000.nii.gz"))
        imgarr, spacing = img
        ps, gt = load_p0(prefix, c)
        d = float(np.mean([g.dice(ps[s], gt[s]) for s in ("R", "L")]))
        percase_p0[c][key] = d
        title = (f"{label}  —  {c}   Dice {d:.3f}  "
                 f"(R {g.dice(ps['R'],gt['R']):.3f}/L {g.dice(ps['L'],gt['L']):.3f})"
                 f"   [agg {agg}]")
        mont = os.path.join(odir, f"{c}_slices.png")
        if not os.path.exists(mont):
            g.render_montage(imgarr, ps, gt, mont, title)
        r3d = os.path.join(odir, f"{c}_3d.png")
        if not os.path.exists(r3d):
            g.render_3d(ps, spacing, r3d, f"{label} — {c} (red=R, blue=L)")
        rows.append({"case": c, "dice": d})
    manifest[key] = {"label": label, "dice": agg, "note": "from-scratch 2D baseline",
                     "cases": rows}
    print(f"[{time.time()-t0:6.1f}s] {key:14s} {len(rows)} cases  agg {agg}")

# ---- P0 comparison grids on representative cases (GT + 4 baselines) ----
sel = json.load(open(os.path.join(g.OUT, "_selection.json")))
reps = sel["reps"]
cdir = os.path.join(g.OUT, "comparison")
os.makedirs(cdir, exist_ok=True)


def best_slice(gt):
    ra = gt["R"].sum(axis=(0, 1)); la = gt["L"].sum(axis=(0, 1))
    score = np.minimum(ra, la)
    return int(np.argmax(score if score.max() else ra + la))


for tag, c in reps:
    imgarr, spacing = g.load_vol(os.path.join(g.CT_DIR, f"{c}_0000.nii.gz"))
    _, gt = load_p0("U-Net", c)
    z = best_slice(gt)
    fg = gt["R"] | gt["L"]
    idx = np.array(np.where(fg[:, :, z]))
    y0, x0 = idx.min(1) - 28; y1, x1 = idx.max(1) + 29
    y0, x0 = max(0, y0), max(0, x0)
    ctsl = g.window_ct(imgarr[:, :, z])[y0:y1, x0:x1]
    panels = ["GT"] + P0_ORDER
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.5),
                             squeeze=False)
    fig.patch.set_facecolor("black")
    for i, key in enumerate(panels):
        ax = axes[0][i]
        ax.set_facecolor("black")
        if key == "GT":
            gt2d = {s: gt[s][y0:y1, x0:x1, z] for s in ("R", "L")}
            g._overlay(ax, ctsl, gt2d, None, None)
            ax.set_title(f"Ground truth (z={z})", fontsize=9.5, fontweight="bold",
                         color="white")
        else:
            ps, _ = load_p0(P0[key][1], c)
            pred2d = {s: ps[s][y0:y1, x0:x1, z] for s in ("R", "L")}
            gt2d = {s: gt[s][y0:y1, x0:x1, z] for s in ("R", "L")}
            g._overlay(ax, ctsl, pred2d, gt2d, None)
            ax.set_title(f"{P0[key][0].replace('P0 ','')}\nDice {percase_p0[c][key]:.3f}"
                         f"  ·  agg {P0[key][2]}", fontsize=8.7, color="white")
    handles = [Patch(facecolor=g.R_COLOR, alpha=0.6, label="Right parotid (pred)"),
               Patch(facecolor=g.L_COLOR, alpha=0.6, label="Left parotid (pred)"),
               Line2D([0], [0], color=g.GT_DASH, lw=1.4, ls=(0, (4, 2)),
                      label="Ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.01), labelcolor="white")
    fig.suptitle(f"From-scratch 2D baselines — {c}  ({tag} case, z={z})",
                 fontsize=12, y=1.0, color="white")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(os.path.join(cdir, f"case_{c}_p0baselines.png"), dpi=200,
                facecolor="black")
    plt.close(fig)
    print(f"  P0 grid {tag:12s} {c}  z={z}")

# write to a separate file to avoid racing render_all's manifest.json
json.dump(manifest, open(os.path.join(g.OUT, "manifest_p0.json"), "w"), indent=1)
json.dump(percase_p0, open(os.path.join(g.OUT, "_percase_p0.json"), "w"))
print(f"\nDone in {time.time()-t0:.1f}s.")
