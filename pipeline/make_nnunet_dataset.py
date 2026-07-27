"""
Convert the reconstructed volumes into an nnU-Net v2 raw dataset for the
parotid model (Phase 1).

Layout produced:
  Dataset<ID>_Parotid/
    imagesTr/ PARxxxx_0000.nii.gz     (train+val patients)
    labelsTr/ PARxxxx.nii.gz          (0=bg, 1=parotid_r, 2=parotid_l)
    imagesTs/ PARxxxx_0000.nii.gz     (locked TEST patients, for final eval)
    labelsTs/ PARxxxx.nii.gz          (kept for OUR one-time test eval; not seen by nnU-Net)
    dataset.json
    case_mapping.json                 (PARxxxx -> real patient id + annotation flags)

Design choices:
  - Only patients with BOTH parotids annotated are included, so the integer
    labelmap is clean (partial-label cases are handled later in Phase 2 via the
    custom masked-loss trainer).
  - The locked test set is written to imagesTs/labelsTs and never enters training.
    nnU-Net does its own 5-fold CV over imagesTr only.
"""
import os, json, argparse
import numpy as np
import nibabel as nib
from build_volumes import reconstruct_patient, _affine

ORGANS = ("PAROTID_R", "PAROTID_L")          # -> label ids 1, 2
LABEL_IDS = {"PAROTID_R": 1, "PAROTID_L": 2}


def _labelmap(recon):
    """Combine per-organ binary vols into one int labelmap (L drawn last)."""
    shape = recon["image"].shape
    lm = np.zeros(shape, dtype=np.uint8)
    for organ in ("PAROTID_R", "PAROTID_L"):
        lm[recon["labels"][organ].astype(bool)] = LABEL_IDS[organ]
    return lm


def _patient_dir(data_dir, pid):
    d = os.path.join(data_dir, pid)
    return d if os.path.isdir(d) else None


def process(pids, data_dir, img_dir, lbl_dir, mapping, start_idx, require_both=True, limit=None):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n, idx, skipped = 0, start_idx, {"absent": 0, "no_recon": 0, "not_both": 0}
    for pid in pids:
        if limit is not None and n >= limit:
            break
        pdir = _patient_dir(data_dir, pid)
        if pdir is None:
            skipped["absent"] += 1; continue
        recon = reconstruct_patient(pdir, ORGANS)
        if recon is None:
            skipped["no_recon"] += 1; continue
        ov = recon["meta"]["organ_voxels"]
        if require_both and (ov["PAROTID_R"] == 0 or ov["PAROTID_L"] == 0):
            skipped["not_both"] += 1; continue

        case = f"PAR{idx:04d}"; idx += 1; n += 1
        aff = _affine(recon["meta"]["z_spacing_mm"])
        nib.save(nib.Nifti1Image(recon["image"], aff), os.path.join(img_dir, f"{case}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(_labelmap(recon), aff), os.path.join(lbl_dir, f"{case}.nii.gz"))
        mapping[case] = {"patient_id": pid, "n_slices": recon["meta"]["n_slices"],
                         "z_spacing_mm": recon["meta"]["z_spacing_mm"],
                         "organ_voxels": ov, "warnings": recon["meta"]["warnings"]}
    return n, idx, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/Users/ritvikmod/Desktop/Projects/ct scan project/ML_Dataset_Final")
    ap.add_argument("--split", default="/Users/ritvikmod/Desktop/Projects/ct scan project/patient classification/dataset_split.json")
    ap.add_argument("--out", required=True, help="output Dataset dir")
    ap.add_argument("--limit", type=int, default=None, help="cap patients per split (for local testing)")
    args = ap.parse_args()

    split = json.load(open(args.split))
    trainval = list(split["train"]) + list(split["val"])
    test = list(split["test"])

    mapping = {}
    n_tr, idx, sk_tr = process(trainval, args.data_dir,
                               os.path.join(args.out, "imagesTr"),
                               os.path.join(args.out, "labelsTr"),
                               mapping, 1, limit=args.limit)
    n_ts, idx, sk_ts = process(test, args.data_dir,
                               os.path.join(args.out, "imagesTs"),
                               os.path.join(args.out, "labelsTs"),
                               mapping, idx, limit=args.limit)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "parotid_r": 1, "parotid_l": 2},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "description": "Bilateral parotid gland segmentation (both-annotated cases).",
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(dataset_json, open(os.path.join(args.out, "dataset.json"), "w"), indent=2)
    json.dump(mapping, open(os.path.join(args.out, "case_mapping.json"), "w"), indent=2)

    print(f"Train+val cases written: {n_tr}   skipped {sk_tr}")
    print(f"Test cases written:      {n_ts}   skipped {sk_ts}")
    print(f"dataset.json numTraining = {n_tr}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
