"""Same-case-across-models comparison grids: one figure per representative case,
the SAME axial slice shown with every model's prediction side by side + GT dashed."""
import os, json
import numpy as np
import galcommon as g
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sel = json.load(open(os.path.join(g.OUT, "_selection.json")))
percase = sel["percase"]
reps = sel["reps"]
cdir = os.path.join(g.OUT, "comparison")
os.makedirs(cdir, exist_ok=True)

PANELS = ["GT"] + g.COMPARE_ORDER   # first panel = ground truth reference


def best_slice(gt, zwin_pad=0):
    """Axial z maximising min(R area, L area) so both glands are visible."""
    R, L = gt["R"], gt["L"]
    ra = R.sum(axis=(0, 1)); la = L.sum(axis=(0, 1))
    score = np.minimum(ra, la)
    if score.max() == 0:
        score = ra + la
    return int(np.argmax(score))


def crop_bbox(gt, pad=28):
    fg = gt["R"] | gt["L"]
    idx = np.array(np.where(fg))
    y0, x0 = idx[0].min(), idx[1].min()
    y1, x1 = idx[0].max(), idx[1].max()
    return (max(0, y0 - pad), min(fg.shape[0], y1 + pad + 1),
            max(0, x0 - pad), min(fg.shape[1], x1 + pad + 1))


manifest_rows = []
for tag, c in reps:
    img, spacing = g.load_vol(os.path.join(g.CT_DIR, f"{c}_0000.nii.gz"))
    gt = g.load_gt(c)
    z = best_slice(gt)
    y0, y1, x0, x1 = crop_bbox(gt)
    ctsl = g.window_ct(img[:, :, z])[y0:y1, x0:x1]

    ncol = 5
    nrow = int(np.ceil(len(PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.3 * nrow),
                             squeeze=False)
    fig.patch.set_facecolor("black")
    for ax in axes.ravel():
        ax.axis("off"); ax.set_facecolor("black")

    for i, key in enumerate(PANELS):
        ax = axes[i // ncol][i % ncol]
        if key == "GT":
            gt2d = {s: gt[s][y0:y1, x0:x1, z] for s in ("R", "L")}
            g._overlay(ax, ctsl, gt2d, None, None)   # GT as filled prediction-style
            ax.set_title(f"Ground truth (z={z})", fontsize=9.5, fontweight="bold",
                         color="white")
        else:
            m = g.MODELS[key]
            pred = g.load_vol(g.P(m["dir"], f"{c}.nii.gz"))[0].astype(np.uint8)
            ps = g.norm_pred(pred, m["remap"])
            pred2d = {s: v[y0:y1, x0:x1, z] for s, v in ps.items()}
            gt2d = {s: gt[s][y0:y1, x0:x1, z] for s in ps}   # GT for predicted sides
            g._overlay(ax, ctsl, pred2d, gt2d, None)
            d = percase[c][key]
            agg = f"  ·  agg {m['dice']}" if m["dice"] else ""
            ax.set_title(f"{m['label']}\nDice {d:.3f}{agg}", fontsize=8.7, color="white")

    handles = [Patch(facecolor=g.R_COLOR, alpha=0.6, label="Right parotid (pred)"),
               Patch(facecolor=g.L_COLOR, alpha=0.6, label="Left parotid (pred)"),
               Line2D([0], [0], color=g.GT_DASH, lw=1.4, ls=(0, (4, 2)),
                      label="Ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.005), labelcolor="white")
    fig.suptitle(f"Model comparison — {c}  ({tag} case, axial z={z})   "
                 f"nnU-Net per-case Dice {percase[c]['nnunet_1fold']:.3f}",
                 fontsize=13, y=0.998, color="white")
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    out = os.path.join(cdir, f"case_{c}_allmodels.png")
    fig.savefig(out, dpi=200, facecolor="black")
    plt.close(fig)
    manifest_rows.append({"case": c, "tag": tag, "z": z,
                          "ref_dice": percase[c]["nnunet_1fold"]})
    print(f"  {tag:12s} {c}  z={z}  -> {os.path.basename(out)}")

json.dump(manifest_rows, open(os.path.join(g.OUT, "_comparison.json"), "w"), indent=1)
print(f"\nWrote {len(manifest_rows)} comparison grids to comparison/")
