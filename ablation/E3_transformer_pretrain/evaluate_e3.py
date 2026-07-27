"""
E3 evaluation — score a trained E3 checkpoint on the LOCKED 43-case test set with
the EXACT Phase-1 3D volumetric protocol used by P0, so pretrained-vs-scratch is
an apples-to-apples comparison (same cases, same metric, same isotropic caveat).

Reuses P0's phase1_metrics. Reads Dataset002 imagesTs (raw pixel) + labelsTs.

Example (pod or CPU):
  python3 evaluate_e3.py --model transunet_pretrained \
      --ckpt ./ckpt_transunet_pretrained/best_model.pth --tag TransUNet_pretrained
"""
import os
import sys
import csv
import argparse
import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ablation_study", "P0_phase1_on_test"))
from phase1_metrics import (calculate_3d_dice, calculate_clinical_tversky,
                            calculate_hd95, calculate_surface_dice, window_and_normalise)
from models_pretrained import build_transunet_pretrained, build_swin_pretrained

DS = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid")
IMAGES_TS, LABELS_TS = os.path.join(DS, "imagesTs"), os.path.join(DS, "labelsTs")


def build(model_name):
    if model_name == "transunet_pretrained":
        return build_transunet_pretrained(pretrained=False, verbose=False)[0]
    if model_name == "swin_pretrained":
        return build_swin_pretrained(pretrained=False, verbose=False)[0]
    raise ValueError(model_name)


@torch.no_grad()
def predict_volume(model, vol, device, batch=8):
    Z = vol.shape[0]
    norm = window_and_normalise(vol)
    pr = np.zeros((Z, 512, 512), np.uint8); pl = np.zeros((Z, 512, 512), np.uint8)
    for s in range(0, Z, batch):
        t = torch.from_numpy(norm[s:s + batch]).unsqueeze(1).to(device)
        p = torch.sigmoid(model(t)).cpu().numpy()
        pr[s:s + batch] = (p[:, 0] > 0.5); pl[s:s + batch] = (p[:, 1] > 0.5)
    return pr, pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["transunet_pretrained", "swin_pretrained"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="row label, e.g. TransUNet_pretrained")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "eval.csv"))
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build(args.model).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    print(f"loaded {args.ckpt} (val_loss {ck.get('val_loss','?')}) on {device}")

    cases = sorted(f[:-7] for f in os.listdir(LABELS_TS) if f.endswith(".nii.gz"))
    assert len(cases) == 43
    D, T, H, S = [], [], [], []
    for i, par in enumerate(cases):
        img = np.transpose(np.asarray(nib.load(f"{IMAGES_TS}/{par}_0000.nii.gz").dataobj), (2, 0, 1)).astype(np.float32)
        lab = np.transpose(np.asarray(nib.load(f"{LABELS_TS}/{par}.nii.gz").dataobj), (2, 0, 1))
        pr, pl = predict_volume(model, img, device)
        for p, t in [(pr, (lab == 1).astype(np.uint8)), (pl, (lab == 2).astype(np.uint8))]:
            D.append(calculate_3d_dice(p, t)); T.append(calculate_clinical_tversky(p, t))
            H.append(calculate_hd95(p, t)); S.append(calculate_surface_dice(p, t))
        print(f"  [{i+1}/43] {par}", flush=True)

    row = [args.tag, f"{np.mean(D):.4f}", f"{np.mean(T):.4f}",
           f"{np.nanmean(H):.2f}", f"{np.mean(S):.4f}", len(D)]
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["Model_Name", "3D_Dice", "Tversky", "HD95_mm", "Surface_Dice_3mm", "n_sides"])
        w.writerow(row)
    print("RESULT:", row)


if __name__ == "__main__":
    main()
