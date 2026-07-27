"""
E7 Part 1 prep — derive parotid annotation presence for the TEST-split patients.

The E2 presence CSV (parotid_presence_trainval.csv) only covers TRAIN+VAL (667 rows).
The locked TEST split (165 patients) was never scanned. This reuses E2's exact
scan logic (same CT-group selection rule as build_volumes) to classify each TEST
patient as both / only_R / only_L / none, so we can identify the single-side test
cases for Part 1.

Output: ablation_study/E7_singleside/parotid_presence_test.csv
"""
import csv, glob, json, os, re, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ML_Dataset_Final"
SPLIT = ROOT / "patient classification/dataset_split.json"
OUT = Path(__file__).resolve().parent / "parotid_presence_test.csv"

ORGANS = ("PAROTID_R", "PAROTID_L")
Z_RE = re.compile(r"Z_(-?\d+\.?\d*)\.npz$")


def _slice_files(pdir):
    groups = {}
    for f in glob.glob(os.path.join(pdir, "**", "*.npz"), recursive=True):
        m = Z_RE.search(os.path.basename(f))
        if not m:
            continue
        grp = os.path.dirname(os.path.dirname(f))
        groups.setdefault(grp, []).append((float(m.group(1)), f))
    return groups


def _group_stats(files):
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
    return {
        "patient": pid, "status": status,
        "n_slices": len(groups[best]), "n_annot_slices": n_annot,
        "vox_R": vr, "vox_L": vl,
        "centroid_col_R": round(colsum["PAROTID_R"] / vr, 1) if vr else "",
        "centroid_col_L": round(colsum["PAROTID_L"] / vl, 1) if vl else "",
    }


def main():
    split = json.load(open(SPLIT))
    test = list(split["test"])
    print(f"scanning {len(test)} TEST patients ...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(scan_patient, test, chunksize=4), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(test)}", flush=True)
    fields = ["patient", "status", "n_slices", "n_annot_slices", "vox_R", "vox_L",
              "centroid_col_R", "centroid_col_L"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    c = Counter(r["status"] for r in rows)
    print("\nTEST-split parotid annotation status:")
    for k in ("both", "only_R", "only_L", "none", "absent", "no_slices"):
        if c.get(k):
            print(f"  {k:10s} {c[k]:4d}")
    print(f"\n  both-parotid                 : {c.get('both', 0)}")
    print(f"  single-side (only_R+only_L)  : {c.get('only_R', 0) + c.get('only_L', 0)}")
    print(f"  none                         : {c.get('none', 0)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
