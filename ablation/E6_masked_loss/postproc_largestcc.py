"""
E4 optional postprocessing — keep only the largest connected component per class
(parotid_r=1, parotid_l=2) in each predicted labelmap, the standard cheap HD95
cleanup (master 15.3). Reads nnU-Net-style labelmaps from --pred-dir, writes cleaned
labelmaps (same names/affine) to --out, ready for pipeline/eval_testset.py.

Expectation: HD95 drops sharply (stray islands removed) with ~unchanged Dice, which
isolates E4's boundary gap to nnU-Net as postprocessing rather than architecture.
"""
import os
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

CONN = generate_binary_structure(3, 3)  # 26-connectivity


def largest_cc(mask):
    lab, n = label(mask, structure=CONN)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (lab == int(np.argmax(sizes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    for f in files:
        nii = nib.load(f)
        lm = np.asarray(nii.dataobj)
        out = np.zeros_like(lm)
        for cls in (1, 2):
            m = (lm == cls)
            if m.any():
                out[largest_cc(m)] = cls
        nib.save(nib.Nifti1Image(out.astype(np.uint8), nii.affine),
                 os.path.join(args.out, os.path.basename(f)))
        print(f"  {os.path.basename(f)}: R {int((lm==1).sum())}->{int((out==1).sum())} "
              f"L {int((lm==2).sum())}->{int((out==2).sum())}", flush=True)
    print(f"wrote {len(files)} cleaned labelmaps -> {args.out}")


if __name__ == "__main__":
    main()
