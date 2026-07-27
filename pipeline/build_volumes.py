"""
Reconstruct 3D volumes from the per-slice .npz dataset.

Each patient folder holds one or more CT groups; each group is a set of
`Z_<pos>.npz` slice files (image + binary masks). This module stacks the slices
in physical order into 3D arrays and writes NIfTI volumes with correct voxel
spacing, ready for nnU-Net / a pretrained 3D model.

Spacing:
  - Z (between-slice) is read from the physical position in the filename.
  - In-plane X/Y is not stored in the npz; we assume the dominant 0.977 mm
    (documented; override via IN_PLANE_MM or a per-patient lookup later).
"""
import os, glob, re, json
import numpy as np
import nibabel as nib

IN_PLANE_MM = 0.977
Z_RE = re.compile(r"Z_(-?\d+\.?\d*)\.npz$")


def _slice_files(pdir):
    """Return {ct_group_dir: [(z, path), ...]} for every CT group in a patient."""
    groups = {}
    for f in glob.glob(os.path.join(pdir, "**", "*.npz"), recursive=True):
        m = Z_RE.search(os.path.basename(f))
        if not m:
            continue
        grp = os.path.dirname(os.path.dirname(f))  # parent of the `data/` dir
        groups.setdefault(grp, []).append((float(m.group(1)), f))
    return groups


def _score_group(files, organs):
    """How many slices in this group contain at least one target organ."""
    n = 0
    for _, f in files:
        try:
            d = np.load(f)
        except Exception:
            continue
        if any(o in d.files and d[o].sum() > 0 for o in organs):
            n += 1
    return n


def reconstruct_patient(pdir, organs=("PAROTID_R", "PAROTID_L")):
    """Build image + label volumes for the best CT group of one patient.

    Returns dict with image (Y,X,Z), labels {organ: (Y,X,Z)}, spacing, meta,
    and a list of warnings — or None if the patient has no usable slices.
    """
    groups = _slice_files(pdir)
    if not groups:
        return None

    # choose the CT group with the most target-organ annotations; tie-break on slice count
    best = max(groups, key=lambda g: (_score_group(groups[g], organs), len(groups[g])))
    files = sorted(groups[best], key=lambda t: t[0])  # sort by physical Z ascending

    warnings = []
    zs = np.array([z for z, _ in files])
    dz = np.diff(zs)
    if len(dz) == 0:
        return None
    z_spacing = float(np.median(dz))
    if not np.allclose(dz, z_spacing, atol=0.05):
        warnings.append(f"non-uniform Z spacing (min={dz.min():.2f} max={dz.max():.2f})")
    # detect gaps (missing slices) relative to the median step
    n_gaps = int(np.sum(np.abs(dz - z_spacing) > 0.5 * z_spacing))
    if n_gaps:
        warnings.append(f"{n_gaps} slice gap(s) detected")
    if np.any(np.abs(dz) < 1e-3):
        warnings.append("duplicate Z position(s)")

    imgs, labels = [], {o: [] for o in organs}
    for _, f in files:
        d = np.load(f)
        imgs.append(d["image"].astype(np.int16))
        for o in organs:
            m = d[o] if o in d.files else np.zeros((512, 512), np.uint8)
            labels[o].append(m.astype(np.uint8))

    image = np.stack(imgs, axis=-1)                       # (Y, X, Z)
    label_vols = {o: np.stack(labels[o], axis=-1) for o in organs}

    meta = {
        "patient_dir": pdir,
        "ct_group": os.path.relpath(best, pdir),
        "n_slices": len(files),
        "z_spacing_mm": z_spacing,
        "in_plane_mm": IN_PLANE_MM,
        "z_range": [float(zs[0]), float(zs[-1])],
        "organ_voxels": {o: int(v.sum()) for o, v in label_vols.items()},
        "warnings": warnings,
    }
    return {"image": image, "labels": label_vols, "meta": meta}


def _affine(z_spacing):
    return np.diag([IN_PLANE_MM, IN_PLANE_MM, z_spacing, 1.0]).astype(np.float32)


def save_nifti(recon, out_prefix):
    """Write <prefix>_0000.nii.gz (image) + <prefix>_<organ>.nii.gz (labels)."""
    aff = _affine(recon["meta"]["z_spacing_mm"])
    nib.save(nib.Nifti1Image(recon["image"], aff), f"{out_prefix}_0000.nii.gz")
    for o, v in recon["labels"].items():
        nib.save(nib.Nifti1Image(v, aff), f"{out_prefix}_{o}.nii.gz")


if __name__ == "__main__":
    import sys
    ROOT = "/Users/ritvikmod/Desktop/Projects/ct scan project"
    DATA = os.path.join(ROOT, "ML_Dataset_Final")
    out = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/claude-501/-Users-ritvikmod-Desktop-Projects-ct-scan-project/c68aec7a-73d9-4844-b1e7-a5a1a7d98913/scratchpad/vol_test"
    os.makedirs(out, exist_ok=True)
    for pid in ["1~240084", "1~240357", "001"]:
        r = reconstruct_patient(os.path.join(DATA, pid))
        if r is None:
            print(f"{pid}: no usable slices"); continue
        save_nifti(r, os.path.join(out, pid.replace("~", "-")))
        print(f"{pid}: img{r['image'].shape} spacing={r['meta']['in_plane_mm']}x{r['meta']['in_plane_mm']}x{r['meta']['z_spacing_mm']} "
              f"organ_vox={r['meta']['organ_voxels']} warn={r['meta']['warnings']}")
