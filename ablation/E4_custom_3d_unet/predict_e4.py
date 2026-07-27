"""
E4 inference — sliding-window prediction of the trained 3D U-Net on the locked
test set, written out as nnU-Net-style labelmaps (0=bg,1=parotid_r,2=parotid_l)
in the SAME geometry as labelsTs, so pipeline/eval_testset.py scores it directly
against the same 43 cases as the nnU-Net 0.8187 baseline.

  python3 predict_e4.py --ckpt ./ckpt_unet3d/best_model.pth \
      --preproc ./preprocessed/test --out ./pred_test
  python3 ../../pipeline/eval_testset.py --pred-dir ./pred_test \
      --gt-dir "../../Parotid-Project/Datasets/Dataset002_Parotid/labelsTs" \
      --model-name Custom_3D_UNet --results-csv ./eval.csv
"""
import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from unet3d import UNet3D
from dataloader_e4 import PATCH

LABELS_TS = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid", "labelsTs")


def gaussian_weight(patch, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in patch]
    g = np.ones(patch, np.float32)
    for d, c in enumerate(coords):
        shape = [1, 1, 1]; shape[d] = patch[d]
        sig = sigma_scale * 2
        g = g * np.exp(-(c ** 2) / (2 * sig ** 2)).reshape(shape)
    return np.clip(g, 1e-3, None).astype(np.float32)


def sliding_window(model, vol, device, patch=PATCH, overlap=0.5, use_amp=True):
    """vol: (Z,Y,X) normalised float. Returns prob (2,Z,Y,X)."""
    Z, Y, X = vol.shape
    # pad volume up to at least patch size
    pad = [(0, max(0, patch[d] - vol.shape[d])) for d in range(3)]
    vpad = np.pad(vol, pad, mode="constant", constant_values=vol.min())
    Zp, Yp, Xp = vpad.shape
    steps = []
    for d, (sz, pz) in enumerate(zip((Zp, Yp, Xp), patch)):
        st = max(1, int(round(pz * (1 - overlap))))
        pos = list(range(0, sz - pz + 1, st))
        if pos[-1] != sz - pz:
            pos.append(sz - pz)
        steps.append(pos)
    gw = gaussian_weight(patch)
    acc = np.zeros((2, Zp, Yp, Xp), np.float32)
    wsum = np.zeros((Zp, Yp, Xp), np.float32)
    model.eval()
    with torch.no_grad():
        for z in steps[0]:
            for y in steps[1]:
                for x in steps[2]:
                    chunk = vpad[z:z+patch[0], y:y+patch[1], x:x+patch[2]]
                    t = torch.from_numpy(chunk[None, None].astype(np.float32)).to(device)
                    if use_amp and device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            logit = model(t)
                        prob = torch.sigmoid(logit.float())[0].cpu().numpy()
                    else:
                        prob = torch.sigmoid(model(t))[0].cpu().numpy()
                    acc[:, z:z+patch[0], y:y+patch[1], x:x+patch[2]] += prob * gw
                    wsum[z:z+patch[0], y:y+patch[1], x:x+patch[2]] += gw
    prob = acc / np.maximum(wsum, 1e-6)
    return prob[:, :Z, :Y, :X]


def to_labelmap(prob, thresh=0.5):
    pr, pl = prob[0], prob[1]
    lm = np.zeros(pr.shape, np.uint8)
    fg = (np.maximum(pr, pl) >= thresh)
    lm[fg & (pr >= pl)] = 1
    lm[fg & (pl > pr)] = 2
    return lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preproc", required=True, help="preprocessed/test dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(1, 2).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    print(f"loaded {args.ckpt} (val {ck.get('val_loss','?')}) on {device}")

    for f in sorted(glob.glob(os.path.join(args.preproc, "*.npz"))):
        par = os.path.basename(f)[:-4]
        vol = np.load(f)["data"].astype(np.float32)             # (Z,Y,X)
        prob = sliding_window(model, vol, device, overlap=args.overlap)
        lm = to_labelmap(prob)                                  # (Z,Y,X)
        lm = np.transpose(lm, (1, 2, 0))                        # -> (Y,X,Z) = labelsTs orient
        aff = nib.load(os.path.join(LABELS_TS, f"{par}.nii.gz")).affine
        nib.save(nib.Nifti1Image(lm, aff), os.path.join(args.out, f"{par}.nii.gz"))
        print(f"  {par}: pred R={int((lm==1).sum())} L={int((lm==2).sum())}", flush=True)
    print(f"wrote predictions -> {args.out}")


if __name__ == "__main__":
    main()
