"""
E3 training-data prep — build a COMPACT dataset.h5 for the 2D transformer training,
containing ONLY the train+val patients and ONLY image + PAROTID_R/L (all other
organs dropped), so the upload to a rented pod is as small as possible.

WHY THIS MATTERS (Track A pod learnings #6): the full ML_Dataset_Final is 55 GB and
the HPC dataset.h5 is 38 GB. Uploading either is ~2.5-3.5 h and ~$8-10 of idle pod
time. This builder drops the 165 test patients (never used for training; the test
set is scored from the nnU-Net NIfTI in E3's eval) and every non-parotid organ.

The output key schema (`<patient>/<CT_group>/data/Z_*.npz`) and contents (image +
PAROTID_R/L) are IDENTICAL to the frozen convert_to_h5.py, so the frozen
hdf5_dataloader.ParotidDataset reads it unchanged and E3 reproduces the Phase-1
2D protocol exactly.

  --cap-empty-per-patient N : optional cost saver. Keep ALL parotid-bearing slices
    but at most N non-parotid slices per patient (random, seeded). This shrinks the
    h5 a lot. NOTE: it slightly changes the pool of negative slices the weighted
    sampler can draw (~22% of training samples are negatives), so it is a mild
    deviation from the exact Phase-1 data -> flag it as a caveat if used. Default 0
    = keep everything (faithful).

Run wherever ML_Dataset_Final is mounted (locally it is present, 55 GB). CPU/IO
bound; this is data-prep, not compute.
"""
import os
import glob
import json
import argparse
import random
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEF_DATA = os.path.join(ROOT, "ML_Dataset_Final")
DEF_SPLIT = os.path.join(ROOT, "patient classification", "dataset_split.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEF_DATA)
    ap.add_argument("--split", default=DEF_SPLIT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val",
                    help="which split lists to include (never 'test')")
    ap.add_argument("--cap-empty-per-patient", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-patients", type=int, default=0, help="validation only")
    args = ap.parse_args()
    assert "test" not in args.splits.split(","), "never put test patients in the training h5"
    random.seed(args.seed)

    split = json.load(open(args.split))
    keep_patients = set()
    for s in args.splits.split(","):
        keep_patients.update(split[s])
    print(f"including splits {args.splits}: {len(keep_patients)} patients")

    files = glob.glob(os.path.join(args.data, "*/*/data/*.npz")) + \
            glob.glob(os.path.join(args.data, "*/data/*.npz"))
    # group by patient
    by_patient = {}
    for f in files:
        pid = f.replace(args.data, "").strip("/").split("/")[0]
        if pid in keep_patients:
            by_patient.setdefault(pid, []).append(f)
    if args.limit_patients:
        by_patient = dict(list(by_patient.items())[:args.limit_patients])
    print(f"matched {len(by_patient)} patients, {sum(len(v) for v in by_patient.values())} slice files")

    n_written = n_par = n_empty = 0
    with h5py.File(args.out, "w") as hf:
        for pi, (pid, pfiles) in enumerate(sorted(by_patient.items())):
            if pi % 25 == 0:
                print(f"  [{pi}/{len(by_patient)}] {pid}", flush=True)
            # split into parotid / empty for optional capping
            par_files, empty_files = [], []
            for f in pfiles:
                try:
                    with np.load(f) as d:
                        has = ("PAROTID_R" in d.files and d["PAROTID_R"].sum() > 0) or \
                              ("PAROTID_L" in d.files and d["PAROTID_L"].sum() > 0)
                except Exception:
                    continue
                (par_files if has else empty_files).append(f)
            if args.cap_empty_per_patient and len(empty_files) > args.cap_empty_per_patient:
                empty_files = random.sample(empty_files, args.cap_empty_per_patient)
            for f in par_files + empty_files:
                key = f.replace(args.data, "").strip("/")
                try:
                    with np.load(f) as d:
                        grp = hf.create_group(key)
                        grp.create_dataset("image", data=d["image"], compression="lzf")
                        if "PAROTID_R" in d.files:
                            grp.create_dataset("PAROTID_R", data=d["PAROTID_R"], compression="lzf")
                        if "PAROTID_L" in d.files:
                            grp.create_dataset("PAROTID_L", data=d["PAROTID_L"], compression="lzf")
                    n_written += 1
                except Exception as e:
                    print(f"  err {f}: {e}")
            n_par += len(par_files); n_empty += len(empty_files)
    sz = os.path.getsize(args.out) / 1e9
    print(f"\nwrote {n_written} slices ({n_par} parotid, {n_empty} empty) -> {args.out} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
