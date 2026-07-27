"""
E7 Part 2 — build the two single-class expert datasets from Dataset003 (ON THE POD).

Dataset005_LeftExpert  = every TRAIN/VAL patient WITH a left parotid  (both 208 + only_L 112 = 320),
                         foreground = left parotid only (label 2 -> 1).
Dataset006_RightExpert = every TRAIN/VAL patient WITH a right parotid (both 208 + only_R 110 = 318),
                         foreground = right parotid only (label 1 -> 1).

Reuses Dataset003_ParotidDirty verbatim: images are HARDLINKED (os.link, instant, byte-exact — same
trick the playbook uses), labels are rewritten to a single foreground class. No reconstruction, no
test leakage (Dataset003 is TRAIN/VAL only, asserted by the expected 320/318 counts).

Run on the pod:  python3 build_experts_pod.py --raw $nnUNet_raw
"""
import os, sys, glob, json, argparse
import numpy as np
import nibabel as nib


def build(src, dst_dir, keep_label, name):
    idir, ldir = f"{dst_dir}/imagesTr", f"{dst_dir}/labelsTr"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    kept = 0
    for lf in sorted(glob.glob(f"{src}/labelsTr/*.nii.gz")):
        case = os.path.basename(lf)[:-7]
        nii = nib.load(lf); L = np.asarray(nii.dataobj)
        if (L == keep_label).sum() == 0:
            continue  # this patient has no gland on this side
        newL = (L == keep_label).astype(np.uint8)  # single foreground class = 1
        nib.save(nib.Nifti1Image(newL, nii.affine), f"{ldir}/{case}.nii.gz")
        img_src = f"{src}/imagesTr/{case}_0000.nii.gz"
        img_dst = f"{idir}/{case}_0000.nii.gz"
        if os.path.exists(img_dst):
            os.remove(img_dst)
        try:
            os.link(img_src, img_dst)          # hardlink (instant, byte-exact)
        except OSError:
            import shutil; shutil.copy2(img_src, img_dst)
        kept += 1
    json.dump({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "parotid": 1},
        "numTraining": kept,
        "file_ending": ".nii.gz",
        "description": f"E7 single-class {name} expert (foreground = {name} parotid only).",
    }, open(f"{dst_dir}/dataset.json", "w"), indent=2)
    print(f"{name}: built {kept} cases -> {dst_dir}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="$nnUNet_raw (contains Dataset003_ParotidDirty)")
    args = ap.parse_args()
    src = f"{args.raw}/Dataset003_ParotidDirty"
    assert os.path.isdir(src), f"missing {src}"

    nL = build(src, f"{args.raw}/Dataset005_LeftExpert", keep_label=2, name="left")
    nR = build(src, f"{args.raw}/Dataset006_RightExpert", keep_label=1, name="right")

    # canaries — the verified TRAIN/VAL composition (both 208 + only_L 112 / only_R 110)
    assert nL == 320, f"LEFT expert expected 320, got {nL}"
    assert nR == 318, f"RIGHT expert expected 318, got {nR}"
    print("OK: left=320 right=318 (matches verified TRAIN/VAL pools; no test leakage)")


if __name__ == "__main__":
    main()
