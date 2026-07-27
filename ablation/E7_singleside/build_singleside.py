"""
E7 Part 1 — build the 58 single-side TEST-split patients into evaluable volumes.

Reuses pipeline/build_volumes.reconstruct_patient (identical CT-group selection +
0.977x0.977xZ affine) and the E6/E4 CTNormalization constants, so the single-side
cases are preprocessed EXACTLY like the clean-43 test set every model was scored on.

For each single-side patient it writes:
  nnraw/imagesTs/<case>_0000.nii.gz   image (Y,X,Z), affine diag(.977,.977,3.0)
  nnraw/labelsTs/<case>.nii.gz        GT labelmap (0=bg,1=parotid_r,2=parotid_l),
                                      single foreground side, label assigned by
                                      GEOMETRY (centroid col < midline -> R=1 else L=2)
                                      to match the convention every model learned.
  e6npz/<case>.npz                    data(Z,Y,X f16, CTNormalized), seg(Z,Y,X u8),
                                      mask[2]  (for the custom E4/E6 predict_e6.py)

Also writes case_map.json: case -> {patient, status, label, centroid_col, z_spacing}.

All patients are TEST-split (verified upstream); NONE were trained on by any model.
"""
import os, sys, json, csv
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from build_volumes import reconstruct_patient, _affine

DATA = os.path.join(ROOT, "ML_Dataset_Final")
PRESENCE = os.path.join(HERE, "parotid_presence_test.csv")
NORM = json.load(open(os.path.join(ROOT, "ablation_study", "E6_masked_loss", "norm_constants.json")))
MID = 255.5  # image is 512 wide; parotids sit well lateral of this


def ct_normalize(x, c):
    x = np.clip(x.astype(np.float32), c["lower"], c["upper"])
    return (x - c["mean"]) / max(c["std"], 1e-8)


def main():
    out_img = os.path.join(HERE, "nnraw", "imagesTs")
    out_lab = os.path.join(HERE, "nnraw", "labelsTs")
    out_npz = os.path.join(HERE, "e6npz")
    for d in (out_img, out_lab, out_npz):
        os.makedirs(d, exist_ok=True)

    rows = list(csv.DictReader(open(PRESENCE)))
    ss = [r for r in rows if r["status"] in ("only_R", "only_L")]
    ss.sort(key=lambda r: r["patient"])
    print(f"building {len(ss)} single-side test patients")

    case_map = {}
    for i, r in enumerate(ss, 1):
        pid = r["patient"]
        rec = reconstruct_patient(os.path.join(DATA, pid))
        if rec is None:
            print(f"  ! {pid}: no usable slices, SKIPPED"); continue
        img = rec["image"]                                  # (Y,X,Z) int16
        mr = rec["labels"]["PAROTID_R"].astype(bool)
        ml = rec["labels"]["PAROTID_L"].astype(bool)
        gland = mr | ml
        assert gland.any(), f"{pid} single-side but empty gland"
        # geometric label: mean column of the annotated gland vs midline
        col = float(np.where(gland)[1].mean())
        label = 1 if col < MID else 2                       # 1=R (lower col), 2=L
        clin = 1 if r["status"] == "only_R" else 2
        if label != clin:
            print(f"  ~ {pid}: geometry(label={label}) != clinician({r['status']}); "
                  f"using GEOMETRY (col={col:.1f})")

        lm = np.zeros(img.shape, np.uint8)
        lm[gland] = label

        case = f"SS{i:04d}"
        zsp = rec["meta"]["z_spacing_mm"]
        aff = _affine(zsp)
        nib.save(nib.Nifti1Image(img.astype(np.int16), aff), os.path.join(out_img, f"{case}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lm, aff), os.path.join(out_lab, f"{case}.nii.gz"))

        # E6 npz: (Z,Y,X)
        data = ct_normalize(np.transpose(img, (2, 0, 1)), NORM).astype(np.float16)
        seg = np.transpose(lm, (2, 0, 1)).astype(np.uint8)
        mask = np.array([1.0 if label == 1 else 0.0, 1.0 if label == 2 else 0.0], np.float32)
        np.savez_compressed(os.path.join(out_npz, f"{case}.npz"), data=data, seg=seg, mask=mask)

        case_map[case] = {"patient": pid, "status": r["status"], "label": label,
                          "centroid_col": round(col, 1), "z_spacing": zsp,
                          "gland_vox": int(gland.sum())}
        if i % 10 == 0 or i == len(ss):
            print(f"  [{i}/{len(ss)}] {case} <- {pid} {r['status']} label={label} "
                  f"vox={int(gland.sum())} shape={img.shape}")

    json.dump(case_map, open(os.path.join(HERE, "case_map.json"), "w"), indent=2)
    nR = sum(1 for v in case_map.values() if v["label"] == 1)
    nL = sum(1 for v in case_map.values() if v["label"] == 2)
    print(f"\nwrote {len(case_map)} cases: R-side={nR} L-side={nL}")
    print(f"  nnraw -> {os.path.join(HERE,'nnraw')}")
    print(f"  e6npz -> {out_npz}")


if __name__ == "__main__":
    main()
