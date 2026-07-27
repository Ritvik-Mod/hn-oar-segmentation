"""
E7 Part 1 — contralateral-suppression montage.

For a few single-side cases, show the CT (axial slice at the annotated gland's
centroid) with each model's prediction overlaid: annotated side (green) and
contralateral/healthy side (red). Illustrates that some arms suppress the
un-annotated gland (annotation-gap artefact) while others also segment it
(anatomically correct, per master 16). Runs anywhere (pod or Mac).
"""
import os, json, argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("nnUNet_clean208", "nnunet_clean208"), ("E4_clean208", "e4"),
          ("E6_dirty430", "dirty430"), ("E6_masked430", "masked430"),
          ("E7_experts", "experts_ss")]


def overlay(ax, ct2d, lm2d, k):
    ax.imshow(ct2d, cmap="gray")
    ck = 3 - k
    rgba = np.zeros((*lm2d.shape, 4))
    rgba[lm2d == k] = [0.2, 1.0, 0.2, 0.55]
    rgba[lm2d == ck] = [1.0, 0.2, 0.2, 0.55]
    ax.imshow(rgba)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=os.path.join(HERE, "preds"))
    ap.add_argument("--img-dir", default=os.path.join(HERE, "nnraw", "imagesTs"))
    ap.add_argument("--gt-dir", default=os.path.join(HERE, "nnraw", "labelsTs"))
    ap.add_argument("--case-map", default=os.path.join(HERE, "case_map.json"))
    ap.add_argument("--cases", nargs="+", default=None,
                    help="case ids; default = auto-pick 2 only_R + 2 only_L")
    ap.add_argument("--out", default=os.path.join(HERE, "contralateral_montage.png"))
    args = ap.parse_args()
    cmap = json.load(open(args.case_map))
    models = [(n, s) for n, s in MODELS
              if os.path.isdir(os.path.join(args.pred_root, s))]

    cases = args.cases
    if not cases:
        rr = [c for c, m in sorted(cmap.items()) if m["status"] == "only_R"][:2]
        ll = [c for c, m in sorted(cmap.items()) if m["status"] == "only_L"][:2]
        cases = rr + ll

    ncol = 1 + len(models)
    fig, axes = plt.subplots(len(cases), ncol, figsize=(2.4 * ncol, 2.6 * len(cases)))
    if len(cases) == 1:
        axes = axes[None, :]

    for ri, case in enumerate(cases):
        meta = cmap[case]; k = meta["label"]
        ct = np.asarray(nib.load(os.path.join(args.img_dir, f"{case}_0000.nii.gz")).dataobj)
        gt = np.asarray(nib.load(os.path.join(args.gt_dir, f"{case}.nii.gz")).dataobj)
        z = int(np.argmax((gt == k).sum(axis=(0, 1))))
        ct2d = ct[:, :, z]
        overlay(axes[ri, 0], ct2d, gt[:, :, z], k)
        axes[ri, 0].set_ylabel(f"{meta['patient']}\n{meta['status']}", fontsize=8)
        if ri == 0:
            axes[ri, 0].set_title("Ground truth", fontsize=9)
        for ci, (mname, sub) in enumerate(models, 1):
            pr = np.asarray(nib.load(os.path.join(args.pred_root, sub, f"{case}.nii.gz")).dataobj)
            overlay(axes[ri, ci], ct2d, pr[:, :, z], k)
            if ri == 0:
                axes[ri, ci].set_title(mname, fontsize=9)

    fig.suptitle("Single-side cases — green = annotated (at-risk) side, "
                 "red = contralateral/healthy gland the model also predicts", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
