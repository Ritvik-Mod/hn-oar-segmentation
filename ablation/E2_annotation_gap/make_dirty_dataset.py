"""
E2 — build the "dirty" nnU-Net dataset (Dataset003_ParotidDirty).

Adapted from `pipeline/make_nnunet_dataset.py` (copied, not edited in place — the original is frozen).

CONSTRUCTION (deliberately: dirty == clean UNION single-side)
  Dataset003 train = the clean Dataset002 training set, cloned byte-for-byte (208 both-annotated
  cases, already L/R-corrected and QC-filtered)  +  the 222 train/val patients that have exactly
  ONE parotid contoured, reconstructed from the npz with their labels AS-IS (the un-contoured
  contralateral gland is left as BACKGROUND).

  Cloning rather than re-reconstructing the 208 guarantees the both-annotated subset is *identical*
  to the clean baseline's, so the ONLY difference between the E2 run and the 0.8187 baseline is the
  222 added partial-label patients. It also saves ~6.5 GB and most of the CPU time.

  This reproduces the annotation-gap noise the Phase-1 baselines trained under (master §16.1).

HELD IDENTICAL
  - The TEST set is NOT rebuilt. imagesTs/labelsTs are cloned from the clean Dataset002 test set
    (the locked 43 QC-clean both-parotid cases). Only the TRAINING condition changes. Golden rule.
  - QC volume filter: the SAME absolute threshold Dataset002 used (3x the median gland volume,
    computed from Dataset002's own labels ~= 18681 voxels) is applied to the 222 new cases, so both
    subsets are filtered by one consistent rule.
  - Geometric L/R correction is applied to the new cases exactly as `fix_labels_qc.py` does. It is a
    no-op on single-gland cases (the function returns early), so they keep their SOURCE L/R naming.
    That was validated safe: all 222/222 single-side glands sit on the anatomically expected side of
    the midline (see RESULT.md). No L/R noise is introduced.

NOTHING ON DISK IS MODIFIED. Source datasets are read-only; images are hardlinked (same filesystem,
instant, no extra space), so the clean Dataset002 is never touched.

CASE IDS: cloned clean cases keep their original ids (PAR0001..PAR0208). New single-side cases are
numbered from PAR1001 so the two groups stay distinguishable in `case_mapping.json`.

Usage:
  python3 make_dirty_dataset.py \
      --data-dir      "<...>/ML_Dataset_Final" \
      --split         "<...>/patient classification/dataset_split.json" \
      --clean-dataset "<...>/Parotid-Project/Datasets/Dataset002_Parotid" \
      --clean-mapping "<...>/Parotid-Project/Datasets/Dataset001_Parotid/case_mapping.json" \
      --out           "<...>/ablation_study/E2_annotation_gap/nnUNet_raw/Dataset003_ParotidDirty"
"""

import argparse
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline"))
from build_volumes import reconstruct_patient, _affine  # noqa: E402

ORGANS = ("PAROTID_R", "PAROTID_L")
LABEL_IDS = {"PAROTID_R": 1, "PAROTID_L": 2}
QC_MULT = 3.0

_ARGS = {}


def _labelmap(recon):
    lm = np.zeros(recon["image"].shape, dtype=np.uint8)
    for organ in ORGANS:
        lm[recon["labels"][organ].astype(bool)] = LABEL_IDS[organ]
    return lm


def relabel_geometric(L):
    """Identical to pipeline/fix_labels_qc.relabel_geometric — no-op when only one gland present."""
    out = np.zeros_like(L)
    r1, r2 = (L == 1), (L == 2)
    if r1.sum() == 0 or r2.sum() == 0:
        return L  # single-side: keep source naming (validated correct for all 222)
    c1 = np.where(r1)[1].mean()
    c2 = np.where(r2)[1].mean()
    if c1 <= c2:
        out[r1] = 1; out[r2] = 2
    else:
        out[r1] = 2; out[r2] = 1
    return out


def clone(src, dst):
    """Hardlink (instant, no extra space, never mutates the source); fall back to copy."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def qc_threshold(clean_dir):
    """3x median gland volume, computed over the clean Dataset002 labels (matches fix_labels_qc)."""
    vols = []
    for lf in glob.glob(f"{clean_dir}/labelsTr/*.nii.gz") + glob.glob(f"{clean_dir}/labelsTs/*.nii.gz"):
        L = np.asarray(nib.load(lf).dataobj)
        for lid in (1, 2):
            v = int((L == lid).sum())
            if v > 0:
                vols.append(v)
    med = float(np.median(vols))
    return QC_MULT * med, med


def build_single(job):
    """Reconstruct one single-side patient -> (case, record) or (case, None) if unusable/QC-dropped."""
    case, pid = job
    out, data_dir, thr = _ARGS["out"], _ARGS["data_dir"], _ARGS["thr"]
    pdir = os.path.join(data_dir, pid)
    if not os.path.isdir(pdir):
        return case, {"patient_id": pid, "status": "absent"}
    recon = reconstruct_patient(pdir, ORGANS)
    if recon is None:
        return case, {"patient_id": pid, "status": "no_recon"}
    ov = recon["meta"]["organ_voxels"]
    if ov["PAROTID_R"] == 0 and ov["PAROTID_L"] == 0:
        return case, {"patient_id": pid, "status": "no_parotid"}
    if max(ov["PAROTID_R"], ov["PAROTID_L"]) > thr:
        return case, {"patient_id": pid, "status": "DROPPED_QC", "organ_voxels": ov}

    L = relabel_geometric(_labelmap(recon))
    aff = _affine(recon["meta"]["z_spacing_mm"])
    nib.save(nib.Nifti1Image(recon["image"], aff), f"{out}/imagesTr/{case}_0000.nii.gz")
    nib.save(nib.Nifti1Image(L.astype(np.uint8), aff), f"{out}/labelsTr/{case}.nii.gz")
    return case, {
        "patient_id": pid,
        "status": "kept",
        "group": "single_side_ADDED",
        "annotation": "only_R" if ov["PAROTID_R"] else "only_L",
        "organ_voxels": ov,
        "n_slices": recon["meta"]["n_slices"],
        "z_spacing_mm": recon["meta"]["z_spacing_mm"],
        "warnings": recon["meta"]["warnings"],
    }


def _init(out, data_dir, thr):
    _ARGS.update(out=out, data_dir=data_dir, thr=thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="ML_Dataset_Final")
    ap.add_argument("--split", required=True, help="dataset_split.json")
    ap.add_argument("--clean-dataset", required=True, help="clean Dataset002_Parotid dir")
    ap.add_argument("--clean-mapping", required=True, help="Dataset001_Parotid/case_mapping.json (PAR->patient_id)")
    ap.add_argument("--out", required=True, help="output Dataset003_ParotidDirty dir")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    for s in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(f"{args.out}/{s}", exist_ok=True)

    mapping = {}

    # --- 1. clone the clean 208 training cases verbatim -----------------------------------------
    m1 = json.load(open(args.clean_mapping))
    clean_tr = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{args.clean_dataset}/labelsTr/*.nii.gz"))
    for case in clean_tr:
        clone(f"{args.clean_dataset}/imagesTr/{case}_0000.nii.gz", f"{args.out}/imagesTr/{case}_0000.nii.gz")
        clone(f"{args.clean_dataset}/labelsTr/{case}.nii.gz", f"{args.out}/labelsTr/{case}.nii.gz")
        mapping[case] = {"patient_id": m1[case]["patient_id"], "status": "kept",
                         "group": "both_annotated_CLONED_from_Dataset002", "annotation": "both"}
    print(f"cloned {len(clean_tr)} clean both-annotated training cases from Dataset002")

    # --- 2. QC threshold, from the clean dataset (one consistent rule) ---------------------------
    thr, med = qc_threshold(args.clean_dataset)
    print(f"QC threshold = {thr:.0f} voxels ({QC_MULT}x Dataset002 median gland {med:.0f})")

    # --- 3. reconstruct the single-side patients -------------------------------------------------
    split = json.load(open(args.split))
    trainval = list(split["train"]) + list(split["val"])
    clean_pids = {m1[c]["patient_id"] for c in clean_tr}
    test_pids = {m1[c]["patient_id"] for c in
                 (os.path.basename(f)[:-7] for f in glob.glob(f"{args.clean_dataset}/labelsTs/*.nii.gz"))}
    # candidates = train/val patients that are not already in the clean set, and never a test patient
    cand = [p for p in trainval if p not in clean_pids and p not in test_pids]
    jobs = [(f"PAR{1000 + i:04d}", pid) for i, pid in enumerate(sorted(cand), 1)]
    print(f"reconstructing {len(jobs)} candidate patients (single-side or no-parotid) ...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.out, args.data_dir, thr)) as ex:
        for i, r in enumerate(ex.map(build_single, jobs, chunksize=2), 1):
            results.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    for case, rec in results:
        mapping[case] = rec

    # --- 4. clone the LOCKED clean test set verbatim ---------------------------------------------
    n_test = 0
    for split_name in ("imagesTs", "labelsTs"):
        for src in sorted(glob.glob(f"{args.clean_dataset}/{split_name}/*.nii.gz")):
            clone(src, f"{args.out}/{split_name}/{os.path.basename(src)}")
            n_test += 1
    print(f"cloned {n_test} files from the LOCKED clean test set (unchanged)")

    # --- 5. dataset.json -------------------------------------------------------------------------
    kept = {c: r for c, r in mapping.items() if r.get("status") == "kept"}
    counts = {"both": 0, "only_R": 0, "only_L": 0}
    for r in kept.values():
        counts[r["annotation"]] += 1
    n_tr = len(kept)

    json.dump({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "parotid_r": 1, "parotid_l": 2},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "description": ("E2 DIRTY partial-label parotid set: clean Dataset002 both-annotated cases "
                        "PLUS every train/val patient with exactly one parotid contoured, labels "
                        "as-is (un-contoured contralateral gland left as background). Test set is "
                        "the clean locked Dataset002 test set, unchanged."),
    }, open(f"{args.out}/dataset.json", "w"), indent=2)
    json.dump(mapping, open(f"{args.out}/case_mapping.json", "w"), indent=2)

    skipped = {}
    for r in mapping.values():
        if r.get("status") != "kept":
            skipped[r["status"]] = skipped.get(r["status"], 0) + 1

    print(f"\nnumTraining = {n_tr}  ({counts['both']} both, {counts['only_R']} only_R, {counts['only_L']} only_L)")
    print(f"  -> single-side (annotation-gap) cases added: {counts['only_R'] + counts['only_L']}")
    print(f"  -> skipped: {skipped}")
    print(f"  -> test cases: {len(glob.glob(f'{args.out}/labelsTs/*.nii.gz'))} (cloned, unchanged)")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
