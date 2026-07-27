"""
E6 preprocessing — reproduce E4's nnU-Net 3d_fullres preprocessing, but for the
DIRTY 430-patient set (Dataset003_ParotidDirty) as the training source, and the
clean locked Dataset002 test set for evaluation. IDENTICAL preprocessing to E4 so
the three arms (clean-208 [=E4], dirty-430, masked-430) differ only in train scope
and loss, never in the pipeline.

Key facts (verified, identical to E4):
  - Dataset003 NIfTI are ALREADY at nnU-Net target spacing [3.0, 0.977, 0.977]
    (affine diag 0.977,0.977,3.0 in (X,Y,Z)) => NO resampling.
  - CTNormalization constants are read from norm_constants.json (the exact
    constants E4 used, from Dataset002's plans.json foreground stats): clip
    [8087, 8394], then (x - 8198.65)/57.49 in raw-pixel space. Reused verbatim so
    dirty/masked-430 are normalised identically to clean-208.

The ONE addition over E4: a per-CASE annotation mask [mask_r, mask_l] stored in
each train npz, derived from the FULL-VOLUME labelmap (mask_r=1 iff label 1 exists
anywhere in that patient, mask_l=1 iff label 2 exists). This is the annotation
status of the whole patient (both / only_R / only_L) — NOT per-patch presence.
The masked-430 arm uses it to ignore an un-annotated gland in the loss; the
dirty-430 arm ignores the mask (treats un-annotated side as penalised background).

Output: one {case}.npz per patient holding
    data float16 (Z,Y,X) normalised, seg uint8 (Z,Y,X) in {0,1,2},
    mask float32 [2] = [mask_r, mask_l].
Also writes case_masks.json (audit trail: case -> [mask_r, mask_l], status).

Run locally (both datasets present on the Mac). CPU, ~minutes.
"""
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# training source: the DIRTY 430-patient set
DS_TRAIN = os.path.join(ROOT, "ablation_study", "E2_annotation_gap", "nnUNet_raw",
                        "Dataset003_ParotidDirty")
# evaluation source: the clean locked test set (unchanged, same 43 cases as all arms)
DS_TEST = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid")
NORM = os.path.join(HERE, "norm_constants.json")


def ct_normalize(x, c):
    x = np.clip(x.astype(np.float32), c["lower"], c["upper"])
    return (x - c["mean"]) / max(c["std"], 1e-8)


def process_train(images_dir, labels_dir, out_dir, c):
    """Dataset003 train: normalise + derive per-case annotation mask from the
    full-volume labelmap."""
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(images_dir)
                   if f.endswith("_0000.nii.gz"))
    audit = {}
    for i, par in enumerate(cases):
        img = np.asarray(nib.load(os.path.join(images_dir, f"{par}_0000.nii.gz")).dataobj)
        img = np.transpose(img, (2, 0, 1))                 # (Z,Y,X)
        data = ct_normalize(img, c).astype(np.float16)
        lab = np.asarray(nib.load(os.path.join(labels_dir, f"{par}.nii.gz")).dataobj)
        seg = np.transpose(lab, (2, 0, 1)).astype(np.uint8)
        # per-CASE mask from the full volume (not per-patch):
        mask_r = 1.0 if (seg == 1).any() else 0.0
        mask_l = 1.0 if (seg == 2).any() else 0.0
        mask = np.array([mask_r, mask_l], np.float32)
        np.savez_compressed(os.path.join(out_dir, f"{par}.npz"),
                            data=data, seg=seg, mask=mask)
        status = ("both" if mask_r and mask_l else
                  "only_R" if mask_r else "only_L" if mask_l else "none")
        audit[par] = {"mask": [mask_r, mask_l], "status": status,
                      "vox_r": int((seg == 1).sum()), "vox_l": int((seg == 2).sum())}
        if i % 25 == 0:
            print(f"  [{i+1}/{len(cases)}] {par} shape={data.shape} "
                  f"status={status} mask={[mask_r, mask_l]}", flush=True)
    json.dump(audit, open(os.path.join(os.path.dirname(out_dir), "case_masks.json"), "w"),
              indent=2)
    n_both = sum(1 for v in audit.values() if v["status"] == "both")
    n_r = sum(1 for v in audit.values() if v["status"] == "only_R")
    n_l = sum(1 for v in audit.values() if v["status"] == "only_L")
    print(f"wrote {len(cases)} train cases -> {out_dir}")
    print(f"  annotation status: both={n_both} only_R={n_r} only_L={n_l} "
          f"(single-side={n_r + n_l}/{len(cases)}={100*(n_r+n_l)/len(cases):.1f}%)")
    return cases


def process_test(images_dir, labels_dir, out_dir, c):
    """Dataset002 test: normalise; masks all-ones (clean both-annotated)."""
    os.makedirs(out_dir, exist_ok=True)
    cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(images_dir)
                   if f.endswith("_0000.nii.gz"))
    for i, par in enumerate(cases):
        img = np.asarray(nib.load(os.path.join(images_dir, f"{par}_0000.nii.gz")).dataobj)
        img = np.transpose(img, (2, 0, 1))
        data = ct_normalize(img, c).astype(np.float16)
        lab = np.asarray(nib.load(os.path.join(labels_dir, f"{par}.nii.gz")).dataobj)
        seg = np.transpose(lab, (2, 0, 1)).astype(np.uint8)
        np.savez_compressed(os.path.join(out_dir, f"{par}.npz"),
                            data=data, seg=seg, mask=np.array([1.0, 1.0], np.float32))
        if i % 20 == 0:
            print(f"  [test {i+1}/{len(cases)}] {par} shape={data.shape}", flush=True)
    print(f"wrote {len(cases)} test cases -> {out_dir}")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "preprocessed"))
    ap.add_argument("--norm", default=NORM)
    ap.add_argument("--limit", type=int, default=0, help="process only N train cases (validation)")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    c = json.load(open(args.norm))
    print("CTNormalization constants (reused from E4):", c)

    train_img = os.path.join(DS_TRAIN, "imagesTr")
    train_lab = os.path.join(DS_TRAIN, "labelsTr")
    if args.limit:
        # validation-only quick path
        cases = sorted(f[:-len("_0000.nii.gz")] for f in os.listdir(train_img)
                       if f.endswith("_0000.nii.gz"))[:args.limit]
        os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
        for par in cases:
            img = np.transpose(np.asarray(nib.load(f"{train_img}/{par}_0000.nii.gz").dataobj), (2, 0, 1))
            seg = np.transpose(np.asarray(nib.load(f"{train_lab}/{par}.nii.gz").dataobj), (2, 0, 1)).astype(np.uint8)
            mask = np.array([1.0 if (seg == 1).any() else 0.0,
                             1.0 if (seg == 2).any() else 0.0], np.float32)
            np.savez_compressed(f"{args.out}/train/{par}.npz",
                                data=ct_normalize(img, c).astype(np.float16), seg=seg, mask=mask)
            print(f"  limited {par} shape={img.shape} mask={mask.tolist()}")
        return

    process_train(train_img, train_lab, os.path.join(args.out, "train"), c)
    if not args.skip_test:
        process_test(os.path.join(DS_TEST, "imagesTs"), os.path.join(DS_TEST, "labelsTs"),
                     os.path.join(args.out, "test"), c)


if __name__ == "__main__":
    main()
