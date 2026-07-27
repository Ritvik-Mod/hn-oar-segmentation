"""
E7 Part 2 — score each expert STANDALONE (not just the combined map), on both test
sets, so we know each expert's own quality and whether combining helps or hurts.

Each expert is single-class (label 1 = its gland). We score it only on cases where
its gland is annotated in GT:
  - Left expert  (Dataset005): both-43 (GT left = label 2) + single-side only_L (28)
  - Right expert (Dataset006): both-43 (GT right = label 1) + single-side only_R (30)
Compared against the combined experts' per-side Dice from the main eval.
"""
import os, sys, json, glob
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from eval_testset import dice, tversky, boundary_metrics

RES = os.path.join(HERE, "_pod_results", "preds")
D2_GT = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid", "labelsTs")
SS_GT = os.path.join(ROOT, "ablation_study", "E7_singleside", "nnraw", "labelsTs")
CASE_MAP = json.load(open(os.path.join(ROOT, "ablation_study", "E7_singleside", "case_map.json")))


def score(pred_dir, gt_dir, gt_label, case_filter=None):
    """Dice/HD95/SurfDice of (pred==1) vs (gt==gt_label), over matching cases."""
    ds, hds, sds = [], [], []
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz"))):
        case = os.path.basename(gt_path)[:-7]
        if case_filter and not case_filter(case):
            continue
        pr_path = os.path.join(pred_dir, f"{case}.nii.gz")
        if not os.path.exists(pr_path):
            continue
        gt = np.asarray(nib.load(gt_path).dataobj)
        pr = np.asarray(nib.load(pr_path).dataobj)
        g = (gt == gt_label).astype(np.uint8)
        if g.sum() == 0:
            continue
        p = (pr == 1).astype(np.uint8)
        hd, sd = boundary_metrics(p, g)
        ds.append(dice(p, g)); hds.append(hd); sds.append(sd)
    return (round(float(np.mean(ds)), 4), round(float(np.nanmean(hds)), 2),
            round(float(np.mean(sds)), 4), len(ds))


def main():
    only_L = lambda c: CASE_MAP.get(c, {}).get("status") == "only_L"
    only_R = lambda c: CASE_MAP.get(c, {}).get("status") == "only_R"
    rows = []
    # expert, set, pred_dir, gt_dir, gt_label, filter
    jobs = [
        ("LEFT expert",  "both-43",        os.path.join(RES, "left_both"),  D2_GT, 2, None),
        ("RIGHT expert", "both-43",        os.path.join(RES, "right_both"), D2_GT, 1, None),
        ("LEFT expert",  "single-side L28", os.path.join(RES, "left_ss"),   SS_GT, 2, only_L),
        ("RIGHT expert", "single-side R30", os.path.join(RES, "right_ss"),  SS_GT, 1, only_R),
    ]
    print(f"{'expert':13s} {'set':17s} {'Dice':>7s} {'HD95':>7s} {'SurfDice':>9s} {'n':>4s}")
    for name, setname, pd, gd, lab, filt in jobs:
        d, hd, sd, n = score(pd, gd, lab, filt)
        print(f"{name:13s} {setname:17s} {d:7.4f} {hd:7.2f} {sd:9.4f} {n:4d}")
        rows.append({"expert": name, "set": setname, "dice": d, "hd95_mm": hd,
                     "surface_dice": sd, "n": n})
    import csv
    with open(os.path.join(HERE, "_pod_results", "results", "individual_experts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nwrote _pod_results/results/individual_experts.csv")


if __name__ == "__main__":
    main()
