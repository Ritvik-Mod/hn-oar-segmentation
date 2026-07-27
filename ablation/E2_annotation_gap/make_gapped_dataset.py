"""
E2b — build the CONSTANT-N "gapped" nnU-Net dataset (Dataset004_ParotidGapped).

WHY THIS EXISTS
  E2 as originally specified changes TWO variables at once: it makes labels dirty AND more than
  doubles the training set (208 -> 430 patients). A null result would then be ambiguous ("the gap is
  free" vs "the gap cost real points but 2x the data compensated").

  E2b isolates the variable properly. It takes the SAME 208 both-annotated patients as the clean
  Dataset002 baseline and *simulates* the annotation gap by deleting one gland's contour on a
  matched fraction of them. N, patient composition, and the images are all identical to the clean
  baseline -- ONLY the label condition differs. That is a true one-variable isolation of the
  annotation gap, and comparing the three arms

      clean-208 (0.8187)  vs  gapped-208 (E2b)  vs  dirty-430 (E2)

  separates "label noise" from "more data", which neither arm gives on its own.

HOW THE GAP IS MATCHED TO REALITY (measured, see RESULT.md)
  In the real train/val pool: 222 of 430 parotid-bearing patients (51.63%) are single-side, split
  110 "only_R" (i.e. the L gland is the one NOT contoured) and 112 "only_L" (R not contoured).
  So E2b gaps round(208 * 0.5163) = 107 of the 208 cases, and within those splits the deleted side
  110:112 -> 53 cases with L deleted, 54 with R deleted. The remaining 101 cases stay both-annotated.
  Selection is seeded (default 42, the project's convention) and recorded per-case, so the dataset
  is exactly reproducible.

SAFETY — NOTHING ON DISK IS MODIFIED
  The source Dataset002 is treated as strictly read-only. Images are HARDLINKED into the new folder
  (instant, no extra space, and a hardlink never mutates the source's contents). Labels are read,
  modified IN MEMORY, and written as NEW files under --out. The original Dataset002 labels are never
  opened for writing. The test set is cloned unchanged.

Usage:
  python3 make_gapped_dataset.py \
      --clean-dataset "<...>/Parotid-Project/Datasets/Dataset002_Parotid" \
      --clean-mapping "<...>/Parotid-Project/Datasets/Dataset001_Parotid/case_mapping.json" \
      --out           "<...>/ablation_study/E2_annotation_gap/nnUNet_raw/Dataset004_ParotidGapped" \
      --seed 42
"""

import argparse
import glob
import json
import os
import random
import shutil

import numpy as np
import nibabel as nib

# Measured from the real data (scan_parotid_presence.py): 222/430 single-side,
# of which 110 are "only_R" (L gland missing) and 112 are "only_L" (R gland missing).
REAL_SINGLE, REAL_TOTAL = 222, 430
REAL_ONLY_R, REAL_ONLY_L = 110, 112

LABEL_R, LABEL_L = 1, 2


def clone(src, dst):
    """Hardlink (instant, no extra space, never mutates the source); fall back to copy."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dataset", required=True, help="clean Dataset002_Parotid dir (READ-ONLY)")
    ap.add_argument("--clean-mapping", required=True, help="Dataset001_Parotid/case_mapping.json")
    ap.add_argument("--out", required=True, help="output Dataset004_ParotidGapped dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for s in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(f"{args.out}/{s}", exist_ok=True)

    m1 = json.load(open(args.clean_mapping))
    cases = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{args.clean_dataset}/labelsTr/*.nii.gz"))
    n = len(cases)

    gap_rate = REAL_SINGLE / REAL_TOTAL
    n_gap = round(n * gap_rate)
    # within the gapped cases, match the real deleted-side ratio (only_R => L deleted)
    n_del_L = round(n_gap * REAL_ONLY_R / REAL_SINGLE)
    n_del_R = n_gap - n_del_L

    print(f"clean cases: {n}")
    print(f"real single-side rate: {REAL_SINGLE}/{REAL_TOTAL} = {gap_rate:.4f}")
    print(f"-> gapping {n_gap} cases ({n_del_L} with L deleted, {n_del_R} with R deleted); "
          f"{n - n_gap} stay both-annotated")

    rng = random.Random(args.seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    to_gap = shuffled[:n_gap]
    delete_L = set(to_gap[:n_del_L])
    delete_R = set(to_gap[n_del_L:])

    mapping, counts = {}, {"both": 0, "only_R": 0, "only_L": 0}
    for case in cases:
        lf = f"{args.clean_dataset}/labelsTr/{case}.nii.gz"
        nii = nib.load(lf)
        L = np.asarray(nii.dataobj).copy()  # copy -> never touch the source array/file
        before = {"R": int((L == LABEL_R).sum()), "L": int((L == LABEL_L).sum())}

        if case in delete_L:
            L[L == LABEL_L] = 0           # contralateral gland left as BACKGROUND
            annotation, deleted = "only_R", "L"
        elif case in delete_R:
            L[L == LABEL_R] = 0
            annotation, deleted = "only_L", "R"
        else:
            annotation, deleted = "both", None
        counts[annotation] += 1

        nib.save(nib.Nifti1Image(L.astype(np.uint8), nii.affine), f"{args.out}/labelsTr/{case}.nii.gz")
        clone(f"{args.clean_dataset}/imagesTr/{case}_0000.nii.gz", f"{args.out}/imagesTr/{case}_0000.nii.gz")
        mapping[case] = {
            "patient_id": m1[case]["patient_id"],
            "annotation": annotation,
            "deleted_side": deleted,
            "voxels_before": before,
            "voxels_after": {"R": int((L == LABEL_R).sum()), "L": int((L == LABEL_L).sum())},
        }

    n_test = 0
    for split_name in ("imagesTs", "labelsTs"):
        for src in sorted(glob.glob(f"{args.clean_dataset}/{split_name}/*.nii.gz")):
            clone(src, f"{args.out}/{split_name}/{os.path.basename(src)}")
            n_test += 1

    json.dump({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "parotid_r": 1, "parotid_l": 2},
        "numTraining": n,
        "file_ending": ".nii.gz",
        "description": (f"E2b GAPPED constant-N parotid set: the same {n} both-annotated patients as "
                        f"clean Dataset002, with one gland's contour deleted on {n_gap} of them "
                        f"(seed {args.seed}) to simulate the measured {gap_rate:.1%} annotation-gap "
                        "rate at constant N. Images identical to Dataset002. Test set unchanged."),
    }, open(f"{args.out}/dataset.json", "w"), indent=2)
    json.dump({"seed": args.seed, "gap_rate": gap_rate, "n_gapped": n_gap,
               "n_deleted_L": n_del_L, "n_deleted_R": n_del_R, "cases": mapping},
              open(f"{args.out}/case_mapping.json", "w"), indent=2)

    print(f"\nnumTraining = {n}  ({counts['both']} both, {counts['only_R']} only_R, {counts['only_L']} only_L)")
    print(f"  -> simulated annotation-gap rate: {(counts['only_R'] + counts['only_L']) / n:.1%} "
          f"(real: {gap_rate:.1%})")
    print(f"  -> test cases cloned unchanged: {len(glob.glob(f'{args.out}/labelsTs/*.nii.gz'))}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
