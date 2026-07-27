"""
E4 preprocessing — reproduce nnU-Net's 3d_fullres preprocessing for Dataset002 so
the custom 3D U-Net trains on IDENTICAL data to the nnU-Net baseline (the whole
point of E4: only the code differs, not the pipeline).

Key facts that make this exact and cheap:
  - The Dataset002 NIfTI volumes are ALREADY at nnU-Net's target spacing
    [3.0, 0.977, 0.977] (verified: affine diag 0.977,0.977,3.0 in (Y,X,Z) =
    (z,y,x) 3.0,0.977,0.977). => NO resampling needed.
  - nnU-Net used CTNormalization with the foreground intensity stats stored in
    plans.json. We read those exact constants and apply the identical transform:
        clip(x, p00_5, p99_5); (x - mean) / std
    (raw-pixel space, because the NIfTI images store raw pixel, not HU — matching
    what nnU-Net itself was fed.)

Output: one {case}.npz per patient under --out, holding
    data float16 (Z,Y,X) normalised, seg uint8 (Z,Y,X) in {0,1,2}.
Also writes norm_constants.json (audit trail) and copies the case list.

Run locally (Dataset002 is present) or on the pod. CPU, ~minutes.
"""
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid")
PLANS = os.path.join(ROOT, "Parotid-Project", "Results",
                     "nnunet_results_Dataset002_noMirror", "Dataset002_Parotid",
                     "nnUNetTrainer_250epochs_noMirror__nnUNetPlans__3d_fullres",
                     "plans.json")


def ct_norm_constants(plans_path=PLANS):
    p = json.load(open(plans_path))
    fg = p["foreground_intensity_properties_per_channel"]["0"]
    return dict(lower=float(fg["percentile_00_5"]), upper=float(fg["percentile_99_5"]),
                mean=float(fg["mean"]), std=float(fg["std"]))


def ct_normalize(x, c):
    x = np.clip(x.astype(np.float32), c["lower"], c["upper"])
    return (x - c["mean"]) / max(c["std"], 1e-8)


def process_split(images_dir, labels_dir, out_dir, c, has_labels=True):
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(images_dir)
                   if f.endswith("_0000.nii.gz"))
    for i, par in enumerate(cases):
        img = np.asarray(nib.load(os.path.join(images_dir, f"{par}_0000.nii.gz")).dataobj)
        img = np.transpose(img, (2, 0, 1))                 # (Z,Y,X)
        data = ct_normalize(img, c).astype(np.float16)
        if has_labels:
            lab = np.asarray(nib.load(os.path.join(labels_dir, f"{par}.nii.gz")).dataobj)
            seg = np.transpose(lab, (2, 0, 1)).astype(np.uint8)
        else:
            seg = np.zeros_like(data, np.uint8)
        np.savez_compressed(os.path.join(out_dir, f"{par}.npz"), data=data, seg=seg)
        if i % 25 == 0:
            print(f"  [{i+1}/{len(cases)}] {par} shape={data.shape} "
                  f"fg_vox={int((seg>0).sum())}", flush=True)
    print(f"wrote {len(cases)} cases -> {out_dir}")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "preprocessed"))
    ap.add_argument("--plans", default=PLANS)
    ap.add_argument("--limit", type=int, default=0, help="process only N train cases (validation)")
    args = ap.parse_args()

    c = ct_norm_constants(args.plans)
    print("CTNormalization constants:", c)
    json.dump(c, open(os.path.join(HERE, "norm_constants.json"), "w"), indent=2)

    tr = process_split(os.path.join(DS, "imagesTr"), os.path.join(DS, "labelsTr"),
                       os.path.join(args.out, "train"), c, has_labels=True) \
        if args.limit == 0 else _limited(DS, args.out, c, args.limit)
    process_split(os.path.join(DS, "imagesTs"), os.path.join(DS, "labelsTs"),
                  os.path.join(args.out, "test"), c, has_labels=True)


def _limited(DS, out, c, n):
    os.makedirs(os.path.join(out, "train"), exist_ok=True)
    imgs = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(os.path.join(DS, "imagesTr"))
                  if f.endswith("_0000.nii.gz"))[:n]
    for par in imgs:
        img = np.transpose(np.asarray(nib.load(f"{DS}/imagesTr/{par}_0000.nii.gz").dataobj), (2, 0, 1))
        lab = np.transpose(np.asarray(nib.load(f"{DS}/labelsTr/{par}.nii.gz").dataobj), (2, 0, 1))
        np.savez_compressed(f"{out}/train/{par}.npz",
                            data=ct_normalize(img, c).astype(np.float16),
                            seg=lab.astype(np.uint8))
        print(f"  limited {par} shape={img.shape}")
    return imgs


if __name__ == "__main__":
    main()
