"""
E7 Part 2 — combine the two single-class expert predictions into a standard
0/1/2 labelmap (right parotid=1, left parotid=2), so it scores with the SAME
pipeline/eval_testset.py (both-parotid 43) and E7_singleside/eval_singleside.py
(single-side 58) as every other model.

Each expert is single-class (foreground=1) and, being trained on one side only,
may also fire on the similar-looking contralateral gland. We resolve by HEMIFIELD:
parotids are strictly lateral, patient-right at LOWER image column, patient-left
at HIGHER (verified on the both-annotated cases). So:
    right-expert (-> label 1)  kept where column  < midline
    left-expert  (-> label 2)  kept where column >= midline
This assigns each expert to its own side (no double-count) yet still captures a
healthy contralateral gland via ITS OWN correct expert — the honest per-side
behaviour we want to measure on the single-side set. Optional largest-CC per class.

  python3 combine_experts.py --right-dir pred_right --left-dir pred_left --out pred_experts [--cc]
"""
import os, glob, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

CONN = generate_binary_structure(3, 3)


def largest_cc(mask):
    lab, n = label(mask, structure=CONN)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(np.argmax(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--right-dir", required=True, help="Dataset006 right-expert preds (0/1)")
    ap.add_argument("--left-dir", required=True, help="Dataset005 left-expert preds (0/1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cc", action="store_true", help="largest-CC per class after combine")
    ap.add_argument("--no-hemifield", action="store_true",
                    help="diagnostic: keep raw expert outputs without hemifield restriction")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for rf in sorted(glob.glob(os.path.join(args.right_dir, "*.nii.gz"))):
        case = os.path.basename(rf)
        lf = os.path.join(args.left_dir, case)
        if not os.path.exists(lf):
            print(f"  ! no left pred for {case}, skipping"); continue
        rn = nib.load(rf); rmask = np.asarray(rn.dataobj) > 0     # (Y,X,Z)
        lmask = np.asarray(nib.load(lf).dataobj) > 0
        X = rmask.shape[1]; mid = X // 2
        col = np.arange(X).reshape(1, X, 1)
        if not args.no_hemifield:
            rmask = rmask & (col < mid)
            lmask = lmask & (col >= mid)
        if args.cc:
            if rmask.any(): rmask = largest_cc(rmask)
            if lmask.any(): lmask = largest_cc(lmask)
        out = np.zeros(rmask.shape, np.uint8)
        out[rmask] = 1
        out[lmask] = 2
        nib.save(nib.Nifti1Image(out, rn.affine), os.path.join(args.out, case))
        print(f"  {case}: R={int((out==1).sum())} L={int((out==2).sum())}", flush=True)
    print(f"wrote combined labelmaps -> {args.out}")


if __name__ == "__main__":
    main()
