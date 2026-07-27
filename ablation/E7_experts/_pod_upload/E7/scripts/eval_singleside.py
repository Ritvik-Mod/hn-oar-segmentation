"""
E7 Part 1 — score every model on the 58 single-side test cases.

Two questions per model:
  (1) ANNOTATED-SIDE accuracy: Dice/Tversky/HD95/SurfaceDice on the ONE annotated
      gland (the at-risk, clinically-relevant side). Reuses pipeline/eval_testset.py
      metrics verbatim. Reported raw and +largest-CC-per-class (to match how the
      E4/E6 baselines were reported), and split by side (R n=30, L n=28).
  (2) CONTRALATERAL-PREDICTION behaviour: does the model ALSO segment the
      un-annotated (healthy) gland? Measured on RAW predictions (pre-postproc):
      contralateral predicted volume, and the % of cases the model "predicts" it.
      Per master 16 this is anatomically CORRECT, not an error.

Outputs: eval_singleside.csv (per-model summary), eval_singleside_percase.csv,
         contralateral_rate.csv, and prints a comparison table.
"""
import os, sys, csv, json, glob, argparse
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# eval_testset.py lives in pipeline/ locally, but alongside this file on the pod:
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from eval_testset import dice, tversky, boundary_metrics
from scipy.ndimage import label, generate_binary_structure

CONN = generate_binary_structure(3, 3)

# absolute floor (voxels) below which a contralateral blob is stray noise, and a
# relative floor vs the ipsilateral prediction, for the "predicts contralateral" flag.
CONTRA_ABS = 500
CONTRA_REL = 0.20

# (model_name, pred_subdir, apply_largest_CC?). experts_ss is already combined+CC'd.
MODELS = [
    ("nnUNet_clean208", "nnunet_clean208", False),
    ("E4_custom_clean208", "e4", True),
    ("E6_dirty430", "dirty430", True),
    ("E6_masked430", "masked430", True),
    ("E7_experts", "experts_ss", False),
]


def largest_cc_per_class(lm):
    out = np.zeros_like(lm)
    for cls in (1, 2):
        m = (lm == cls)
        if not m.any():
            continue
        lab, n = label(m, structure=CONN)
        if n <= 1:
            out[m] = cls
        else:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            out[lab == int(np.argmax(sizes))] = cls
    return out


def contra_largest_blob(mask):
    if not mask.any():
        return 0
    lab, n = label(mask, structure=CONN)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return int(sizes.max()) if len(sizes) > 1 else int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=os.path.join(HERE, "preds"),
                    help="dir containing per-model prediction subdirs")
    ap.add_argument("--gt-dir", default=os.path.join(HERE, "nnraw", "labelsTs"))
    ap.add_argument("--case-map", default=os.path.join(HERE, "case_map.json"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    pred_root, gt_dir, out_dir = args.pred_root, args.gt_dir, args.out_dir
    case_map = json.load(open(args.case_map))

    percase_rows = []
    summary_rows = []
    contra_rows = []

    for model_name, pred_sub, do_cc in MODELS:
        pred_dir = os.path.join(pred_root, pred_sub)
        if not os.path.isdir(pred_dir) or not glob.glob(os.path.join(pred_dir, "*.nii.gz")):
            print(f"!! no predictions for {model_name} at {pred_dir} — skipping")
            continue

        # ipsi metrics accumulators, split by side
        acc = {s: {"dice": [], "tv": [], "hd": [], "sd": [], "dice_cc": []} for s in ("R", "L")}
        contra_vox_list, contra_blob_list, ipsi_pred_vox_list = [], [], []
        n_contra = 0
        n = 0

        for case, meta in sorted(case_map.items()):
            gt_path = os.path.join(gt_dir, f"{case}.nii.gz")
            pr_path = os.path.join(pred_dir, f"{case}.nii.gz")
            if not os.path.exists(pr_path):
                print(f"  ! {model_name}: no pred for {case}"); continue
            gt = np.asarray(nib.load(gt_path).dataobj)
            pr_raw = np.asarray(nib.load(pr_path).dataobj)
            k = meta["label"]          # annotated (ipsilateral) label
            ck = 3 - k                 # contralateral label
            side = "R" if k == 1 else "L"
            n += 1

            g = (gt == k).astype(np.uint8)
            p_raw = (pr_raw == k).astype(np.uint8)
            pr_cc = largest_cc_per_class(pr_raw) if do_cc else pr_raw
            p_cc = (pr_cc == k).astype(np.uint8)

            hd_v, sd_v = boundary_metrics(p_cc, g)
            d_raw = dice(p_raw, g)
            d_cc = dice(p_cc, g)
            tv = tversky(p_cc, g)
            acc[side]["dice"].append(d_raw)
            acc[side]["dice_cc"].append(d_cc)
            acc[side]["tv"].append(tv)
            acc[side]["hd"].append(hd_v)
            acc[side]["sd"].append(sd_v)

            # contralateral (RAW): total voxels + largest blob
            contra_mask_raw = (pr_raw == ck)
            cvox = int(contra_mask_raw.sum())
            cblob = contra_largest_blob(contra_mask_raw)
            ipsi_pred_vox = int(p_raw.sum())
            contra_vox_list.append(cvox)
            contra_blob_list.append(cblob)
            ipsi_pred_vox_list.append(ipsi_pred_vox)
            predicts_contra = (cblob >= CONTRA_ABS) and (
                ipsi_pred_vox == 0 or cblob >= CONTRA_REL * ipsi_pred_vox)
            n_contra += int(predicts_contra)

            percase_rows.append({
                "model": model_name, "case": case, "patient": meta["patient"],
                "status": meta["status"], "side": side,
                "dice_raw": round(d_raw, 4), "dice_cc": round(d_cc, 4),
                "tversky": round(tv, 4),
                "hd95_mm": ("" if np.isnan(hd_v) else round(hd_v, 2)),
                "surface_dice": round(sd_v, 4),
                "ipsi_pred_vox": ipsi_pred_vox, "contra_vox": cvox,
                "contra_blob": cblob, "predicts_contra": int(predicts_contra),
            })

        # aggregate ipsi (annotated-side) across both sides
        def allkey(key):
            vals = acc["R"][key] + acc["L"][key]
            fn = np.nanmean if key == "hd" else np.mean
            return float(fn(vals)) if vals else float("nan")

        d_raw_all = allkey("dice"); d_cc_all = allkey("dice_cc")
        summary_rows.append({
            "model": model_name, "n": n,
            "annot_dice_raw": round(d_raw_all, 4),
            "annot_dice_cc": round(d_cc_all, 4),
            "annot_tversky": round(allkey("tv"), 4),
            "annot_hd95_mm": round(allkey("hd"), 2),
            "annot_surface_dice": round(allkey("sd"), 4),
            "R_dice_cc": round(np.mean(acc["R"]["dice_cc"]), 4) if acc["R"]["dice_cc"] else float("nan"),
            "L_dice_cc": round(np.mean(acc["L"]["dice_cc"]), 4) if acc["L"]["dice_cc"] else float("nan"),
        })
        contra_rows.append({
            "model": model_name, "n": n,
            "mean_contra_vox": round(float(np.mean(contra_vox_list)), 1),
            "median_contra_blob": int(np.median(contra_blob_list)),
            "mean_ipsi_pred_vox": round(float(np.mean(ipsi_pred_vox_list)), 1),
            "n_predicts_contra": n_contra,
            "contra_rate_pct": round(100.0 * n_contra / n, 1) if n else 0.0,
        })

    # write CSVs
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_singleside_percase.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(percase_rows[0].keys()))
        w.writeheader(); w.writerows(percase_rows)
    with open(os.path.join(out_dir, "eval_singleside.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(os.path.join(out_dir, "contralateral_rate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contra_rows[0].keys()))
        w.writeheader(); w.writerows(contra_rows)

    print("\n================ ANNOTATED-SIDE (n=58 single-side) ================")
    print(f"{'model':22s} {'Dice_raw':>9s} {'Dice+CC':>8s} {'Tversky':>8s} {'HD95':>7s} {'SurfDice':>9s} {'R':>7s} {'L':>7s}")
    for r in summary_rows:
        print(f"{r['model']:22s} {r['annot_dice_raw']:9.4f} {r['annot_dice_cc']:8.4f} "
              f"{r['annot_tversky']:8.4f} {r['annot_hd95_mm']:7.2f} {r['annot_surface_dice']:9.4f} "
              f"{r['R_dice_cc']:7.4f} {r['L_dice_cc']:7.4f}")
    print("\n================ CONTRALATERAL-PREDICTION (raw preds) ================")
    print(f"threshold: contra largest-blob >= {CONTRA_ABS} vox AND >= {CONTRA_REL:.0%} of ipsi pred")
    print(f"{'model':22s} {'mean_contra_vox':>15s} {'med_blob':>9s} {'mean_ipsi_vox':>13s} {'contra_rate':>12s}")
    for r in contra_rows:
        print(f"{r['model']:22s} {r['mean_contra_vox']:15.1f} {r['median_contra_blob']:9d} "
              f"{r['mean_ipsi_pred_vox']:13.1f} {r['contra_rate_pct']:11.1f}%")
    print("\nwrote eval_singleside.csv, eval_singleside_percase.csv, contralateral_rate.csv")


if __name__ == "__main__":
    main()
