"""
E2 prep — measure the annotation gap in the TRAIN/VAL pool.

Answers the question the track file asks for: *how many extra (single-side) patients does the
"dirty" dataset add over the clean both-annotated Dataset002?*

This is pure measurement on the existing npz data (`ML_Dataset_Final`, read-only). No training,
no dataset is written. It replicates `build_volumes.reconstruct_patient`'s CT-group selection rule
(pick the group with the most target-organ annotations; tie-break on slice count) so the counts
match what `make_nnunet_dataset.py` would actually produce, but WITHOUT reconstructing the full
image volumes -- only the parotid masks are touched, which makes it ~100x cheaper.

Also records, for each single-side patient, the annotated gland's mean column (axis-1) centroid
and the image midline. That lets us check whether the source L/R *naming* is trustworthy for
single-side cases -- important for E2's validity, because `fix_labels_qc.relabel_geometric()`
cannot fix L/R on a single-gland case (it returns early when one side is empty), so those cases
keep whatever the clinician named them.

Output: ablation_study/E2_annotation_gap/parotid_presence_trainval.csv
"""

import csv
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ML_Dataset_Final"
SPLIT = ROOT / "patient classification/dataset_split.json"
OUT = Path(__file__).resolve().parent / "parotid_presence_trainval.csv"

ORGANS = ("PAROTID_R", "PAROTID_L")
Z_RE = re.compile(r"Z_(-?\d+\.?\d*)\.npz$")


def _slice_files(pdir):
    """{ct_group_dir: [(z, path), ...]} — mirrors build_volumes._slice_files."""
    groups = {}
    for f in glob.glob(os.path.join(pdir, "**", "*.npz"), recursive=True):
        m = Z_RE.search(os.path.basename(f))
        if not m:
            continue
        grp = os.path.dirname(os.path.dirname(f))
        groups.setdefault(grp, []).append((float(m.group(1)), f))
    return groups


def _group_stats(files):
    """(n_slices_with_organ, vox_R, vox_L, col_sum_R, col_sum_L) for one CT group."""
    n_annot = 0
    vox = {o: 0 for o in ORGANS}
    colsum = {o: 0.0 for o in ORGANS}
    for _, f in files:
        try:
            d = np.load(f)
        except Exception:
            continue
        hit = False
        for o in ORGANS:
            if o in d.files:
                m = d[o]
                s = int(m.sum())
                if s > 0:
                    hit = True
                    vox[o] += s
                    # accumulate column (axis-1) positions to get a centroid later
                    colsum[o] += float(np.where(m)[1].sum())
        n_annot += hit
    return n_annot, vox, colsum


def scan_patient(pid):
    pdir = DATA / pid
    if not pdir.is_dir():
        return {"patient": pid, "status": "absent"}
    groups = _slice_files(str(pdir))
    if not groups:
        return {"patient": pid, "status": "no_slices"}

    # replicate build_volumes' group choice: most organ-annotated slices, tie-break slice count
    stats = {g: _group_stats(f) for g, f in groups.items()}
    best = max(groups, key=lambda g: (stats[g][0], len(groups[g])))
    n_annot, vox, colsum = stats[best]

    vr, vl = vox["PAROTID_R"], vox["PAROTID_L"]
    if vr > 0 and vl > 0:
        status = "both"
    elif vr > 0:
        status = "only_R"
    elif vl > 0:
        status = "only_L"
    else:
        status = "none"

    row = {
        "patient": pid,
        "status": status,
        "n_slices": len(groups[best]),
        "n_annot_slices": n_annot,
        "vox_R": vr,
        "vox_L": vl,
        # mean column centroid of each annotated gland (image is 512 wide -> midline 255.5)
        "centroid_col_R": round(colsum["PAROTID_R"] / vr, 1) if vr else "",
        "centroid_col_L": round(colsum["PAROTID_L"] / vl, 1) if vl else "",
    }
    return row


def main():
    split = json.load(open(SPLIT))
    trainval = list(split["train"]) + list(split["val"])
    if len(sys.argv) > 1:  # optional cap, for timing
        trainval = trainval[: int(sys.argv[1])]

    print(f"scanning {len(trainval)} train+val patients ...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(scan_patient, trainval, chunksize=4), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(trainval)}", flush=True)

    fields = ["patient", "status", "n_slices", "n_annot_slices", "vox_R", "vox_L",
              "centroid_col_R", "centroid_col_L"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print("\nTRAIN+VAL parotid annotation status:")
    for k in ("both", "only_R", "only_L", "none", "absent", "no_slices"):
        if c.get(k):
            print(f"  {k:10s} {c[k]:4d}")
    single = c.get("only_R", 0) + c.get("only_L", 0)
    print(f"\n  clean (both-annotated only, = Dataset002 pool) : {c.get('both', 0)}")
    print(f"  dirty (>=1 parotid)                            : {c.get('both', 0) + single}")
    print(f"  EXTRA single-side patients the dirty set adds   : {single}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
