"""
P0 — Re-evaluate the four Phase-1 from-scratch checkpoints on the LOCKED 43-case
test set, using the Phase-1 3D volumetric protocol (stack 2D predictions per
patient, threshold 0.5, per-side Dice/Tversky/HD95/Surface-Dice).

WHY: the four models (U-Net, Attention U-Net, TransUNet, Swin-UNet) were only
ever scored on the VALIDATION set. The whole ablation study is compared on the
same locked TEST set (Dataset002_Parotid/labelsTs = 43 both-parotid, QC-clean
cases). This produces the single comparable "test" column for the four baselines.

DATA SOURCE (deliberate): we evaluate on the Dataset002 reconstructed NIfTI
volumes (imagesTs = raw-pixel int16, labelsTs = 1:parotid_r 2:parotid_l). This is
the EXACT same 43 cases and the EXACT same ground-truth voxels that nnU-Net's
0.8187 was scored on, so Dice is directly comparable across Phase-1 and Phase-2.
The images are raw pixel values (verified min 0 / max ~27k), matching what the
Phase-1 models were trained on, so `window_and_normalise` applies unchanged.

CAVEAT (reported, not hidden): HD95 and Surface-Dice here use the Phase-1
ISOTROPIC 0.977 mm spacing (2D-stacked convention). nnU-Net's HD95/Surf-Dice use
TRUE anisotropic (0.977, 0.977, 3.0). So Dice/Tversky are cross-comparable but the
Phase-1 vs Phase-2 boundary numbers are NOT. See master doc 12.3 / 14.5.

Prediction convention: model logits are (B, 2, H, W); channel 0 = PAROTID_R,
channel 1 = PAROTID_L (matches the dataloader, master 9). sigmoid > 0.5.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import csv
import json
import time
import argparse
import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from phase1_metrics import (
    calculate_3d_dice, calculate_clinical_tversky, calculate_hd95,
    calculate_surface_dice, window_and_normalise, PIXEL_SPACING_MM,
)

# frozen model definitions (read-only originals)
DT = os.path.join(ROOT, "dicom test")
for sub in ["unet_codes_from_hpc", "attention_unet_codes_from_hpc",
            "swinunet_codes_from_hpc", "transunet_codes_from_hpc"]:
    sys.path.insert(0, os.path.join(DT, sub))

from unet import UNet
from attention_unet import AttentionUNet
from swin_unet import SwinUNet
from transunet import TransUNet

DATASET = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid")
IMAGES_TS = os.path.join(DATASET, "imagesTs")
LABELS_TS = os.path.join(DATASET, "labelsTs")
CASE_MAP = os.path.join(ROOT, "Parotid-Project", "Datasets",
                        "Dataset001_Parotid", "case_mapping.json")

MODELS = {
    "U-Net": (UNet, os.path.join(ROOT, "checkpoints", "unet", "best_model_unet.pth")),
    "Attention_UNet": (AttentionUNet, os.path.join(ROOT, "checkpoints", "attention_unet", "best_model_attention.pth")),
    "TransUNet": (TransUNet, os.path.join(ROOT, "checkpoints", "transunet", "best_model_transunet.pth")),
    "Swin_UNet": (SwinUNet, os.path.join(ROOT, "checkpoints", "swin_unet", "best_model_swinunet.pth")),
}


def load_model(cls, ckpt_path, device):
    model = cls(in_channels=1, out_channels=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    return model, epoch, val_loss


@torch.no_grad()
def predict_volume(model, image_vol, device, batch=8):
    """image_vol: (Z, 512, 512) raw pixel. Returns pred_r, pred_l as (Z,512,512) uint8."""
    Z = image_vol.shape[0]
    norm = window_and_normalise(image_vol)  # (Z,512,512) float32 in [0,1]
    pred_r = np.zeros((Z, 512, 512), np.uint8)
    pred_l = np.zeros((Z, 512, 512), np.uint8)
    for s in range(0, Z, batch):
        chunk = norm[s:s + batch]
        t = torch.from_numpy(chunk).unsqueeze(1).to(device)  # (b,1,512,512)
        logits = model(t)
        prob = torch.sigmoid(logits).cpu().numpy()           # (b,2,512,512)
        pred_r[s:s + batch] = (prob[:, 0] > 0.5).astype(np.uint8)
        pred_l[s:s + batch] = (prob[:, 1] > 0.5).astype(np.uint8)
    return pred_r, pred_l


def eval_model(name, model, cases, case_map, device, per_case_writer, preds_dir=None):
    dice_scores, tversky_scores, hd95_scores, sdice_scores = [], [], [], []
    for i, par in enumerate(cases):
        img = np.asarray(nib.load(os.path.join(IMAGES_TS, f"{par}_0000.nii.gz")).dataobj)
        lab = np.asarray(nib.load(os.path.join(LABELS_TS, f"{par}.nii.gz")).dataobj)
        # (512,512,Z) -> (Z,512,512)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        lab = np.transpose(lab, (2, 0, 1))
        gt_r = (lab == 1).astype(np.uint8)
        gt_l = (lab == 2).astype(np.uint8)

        pred_r, pred_l = predict_volume(model, img, device)

        if preds_dir is not None:
            # save compressed predictions (+GT) for reuse by A1 (avoids re-inference)
            np.savez_compressed(
                os.path.join(preds_dir, f"{name}__{par}.npz"),
                pred_r=pred_r, pred_l=pred_l, gt_r=gt_r, gt_l=gt_l)

        row = {"model": name, "case": par,
               "patient_id": case_map.get(par, {}).get("patient_id", "")}
        for side, p, t in [("R", pred_r, gt_r), ("L", pred_l, gt_l)]:
            d = calculate_3d_dice(p, t)
            tv = calculate_clinical_tversky(p, t)
            hd = calculate_hd95(p, t)
            sd = calculate_surface_dice(p, t)
            dice_scores.append(d); tversky_scores.append(tv)
            hd95_scores.append(hd); sdice_scores.append(sd)
            row[f"dice_{side}"] = f"{d:.4f}"
            row[f"tversky_{side}"] = f"{tv:.4f}"
            row[f"hd95_{side}"] = f"{hd:.4f}" if not np.isnan(hd) else "nan"
            row[f"sdice_{side}"] = f"{sd:.4f}"
        per_case_writer.writerow(row)
        print(f"  [{i+1}/{len(cases)}] {par}: DiceR={row['dice_R']} DiceL={row['dice_L']}", flush=True)

    return {
        "3D_Dice": float(np.mean(dice_scores)),
        "Tversky": float(np.mean(tversky_scores)),
        "HD95_mm": float(np.nanmean(hd95_scores)),
        "Surface_Dice_3mm": float(np.mean(sdice_scores)),
        "n_sides": len(dice_scores),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--only", default=None, help="comma-sep subset of model names")
    ap.add_argument("--save-preds", action="store_true",
                    help="save compressed prediction volumes to preds/ for A1 reuse")
    args = ap.parse_args()

    preds_dir = None
    if args.save_preds:
        preds_dir = os.path.join(HERE, "preds")
        os.makedirs(preds_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    cases = sorted(f[:-len(".nii.gz")] for f in os.listdir(LABELS_TS) if f.endswith(".nii.gz"))
    assert len(cases) == 43, f"expected 43 locked test cases, found {len(cases)}"
    print(f"Locked test cases: {len(cases)} ({cases[0]}..{cases[-1]})")
    case_map = json.load(open(CASE_MAP))

    wanted = MODELS if not args.only else {k: MODELS[k] for k in args.only.split(",")}

    summary = []
    per_case_path = os.path.join(HERE, "per_case.csv")
    fieldnames = ["model", "case", "patient_id",
                  "dice_R", "tversky_R", "hd95_R", "sdice_R",
                  "dice_L", "tversky_L", "hd95_L", "sdice_L"]
    with open(per_case_path, "w", newline="") as pcf:
        pcw = csv.DictWriter(pcf, fieldnames=fieldnames)
        pcw.writeheader()
        for name, (cls, ckpt) in wanted.items():
            print(f"\n=== {name} ===", flush=True)
            t0 = time.time()
            model, epoch, val_loss = load_model(cls, ckpt, device)
            print(f"loaded {name} (epoch {epoch}, val_loss {val_loss})", flush=True)
            res = eval_model(name, model, cases, case_map, device, pcw, preds_dir)
            res["model"] = name
            res["epoch"] = epoch
            res["wall_s"] = round(time.time() - t0, 1)
            summary.append(res)
            print(f"{name}: Dice={res['3D_Dice']:.4f} Tversky={res['Tversky']:.4f} "
                  f"HD95={res['HD95_mm']:.2f}mm SurfDice={res['Surface_Dice_3mm']:.4f} "
                  f"({res['wall_s']}s)", flush=True)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    out_csv = os.path.join(HERE, "eval.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model_Name", "3D_Dice", "Tversky", "HD95_mm",
                    "Surface_Dice_3mm", "n_sides", "best_epoch", "wall_s"])
        for r in summary:
            w.writerow([r["model"], f"{r['3D_Dice']:.4f}", f"{r['Tversky']:.4f}",
                        f"{r['HD95_mm']:.2f}", f"{r['Surface_Dice_3mm']:.4f}",
                        r["n_sides"], r["epoch"], r["wall_s"]])
    print(f"\nWrote {out_csv}")
    print(f"Wrote {per_case_path}")
    print("\nSUMMARY (locked 43-case test set, Phase-1 isotropic protocol):")
    for r in summary:
        print(f"  {r['model']:16s} Dice {r['3D_Dice']:.4f} | Tversky {r['Tversky']:.4f} "
              f"| HD95 {r['HD95_mm']:.2f} | SurfDice {r['Surface_Dice_3mm']:.4f}")


if __name__ == "__main__":
    main()
