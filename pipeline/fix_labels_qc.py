"""
Produce a corrected nnU-Net dataset from an existing one:
  1. Consistent L/R naming — reassign the two parotid labels by geometry
     (smaller axis-1 centroid = R = label 1, larger = L = label 2). This fixes
     the ~25% of patients whose source PAROTID_L/R were flipped, WITHOUT touching
     the actual contour shapes (only the label id is swapped).
  2. QC — drop any case whose gland volume exceeds 3x the dataset median
     (catches corrupt/blown-up masks, e.g. PAR0240).

Images are cloned (APFS copy-on-write via `cp -c`, instant, no extra disk).
Usage:  python fix_labels_qc.py <src_dataset_dir> <dst_dataset_dir>
"""
import os, sys, glob, json, shutil
import numpy as np
import nibabel as nib

QC_MULT = 3.0


def gland_vols(L):
    return int((L == 1).sum()), int((L == 2).sum())


def relabel_geometric(L):
    """Swap label ids so smaller axis-1 centroid -> 1 (R), larger -> 2 (L)."""
    out = np.zeros_like(L)
    r1, r2 = (L == 1), (L == 2)
    if r1.sum() == 0 or r2.sum() == 0:
        return L  # only one gland; leave as-is (shouldn't happen in both-annotated)
    c1 = np.where(r1)[1].mean()
    c2 = np.where(r2)[1].mean()
    if c1 <= c2:
        out[r1] = 1; out[r2] = 2
    else:
        out[r1] = 2; out[r2] = 1   # flipped -> correct it
    return out


def clone(src, dst):
    """Hardlink (instant, no extra space, same filesystem); fall back to copy."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(src, dst):
    # pass 1: global median gland volume for the QC threshold
    allv = []
    for f in glob.glob(f"{src}/labelsTr/*.nii.gz") + glob.glob(f"{src}/labelsTs/*.nii.gz"):
        allv += list(gland_vols(np.asarray(nib.load(f).dataobj)))
    thr = QC_MULT * np.median(allv)
    print(f"QC threshold = {thr:.0f} voxels ({QC_MULT}x median {np.median(allv):.0f})")

    mapping = {}
    counts = {}
    for split in ("Tr", "Ts"):
        idir, ldir = f"{dst}/images{split}", f"{dst}/labels{split}"
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        kept = dropped = flipped = 0
        for lf in sorted(glob.glob(f"{src}/labels{split}/*.nii.gz")):
            case = os.path.basename(lf)[:-7]
            nii = nib.load(lf); L = np.asarray(nii.dataobj)
            v1, v2 = gland_vols(L)
            if v1 > thr or v2 > thr:
                dropped += 1; mapping[case] = {"status": "DROPPED_QC", "vox": [v1, v2]}
                continue
            newL = relabel_geometric(L)
            was_flipped = not np.array_equal(newL, L)
            flipped += was_flipped
            nib.save(nib.Nifti1Image(newL.astype(np.uint8), nii.affine), f"{ldir}/{case}.nii.gz")
            clone(f"{src}/images{split}/{case}_0000.nii.gz", f"{idir}/{case}_0000.nii.gz")
            kept += 1
            mapping[case] = {"status": "kept", "lr_flipped": bool(was_flipped)}
        counts[split] = kept
        print(f"  {split}: kept {kept} | dropped {dropped} | L/R-corrected {flipped}")

    json.dump({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "parotid_r": 1, "parotid_l": 2},
        "numTraining": counts["Tr"],
        "file_ending": ".nii.gz",
        "description": "Bilateral parotid; L/R made geometrically consistent + corrupt masks QC-dropped.",
    }, open(f"{dst}/dataset.json", "w"), indent=2)
    json.dump(mapping, open(f"{dst}/case_mapping_fix.json", "w"), indent=2)
    print(f"numTraining={counts['Tr']}  written to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
