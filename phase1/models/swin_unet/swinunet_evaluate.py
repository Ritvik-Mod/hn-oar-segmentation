import os
import torch
import numpy as np
import h5py
import json
import csv
from swin_unet import SwinUNet
from scipy.ndimage import distance_transform_edt, binary_erosion

PIXEL_SPACING_MM = 0.977

def window_and_normalise(image):
    hu_intercept = -8192.0
    hu_min = -150.0
    hu_max = 250.0
    image_hu = image.astype(np.float32) * 1.0 + hu_intercept
    image_clipped = np.clip(image_hu, hu_min, hu_max)
    return (image_clipped - hu_min) / (hu_max - hu_min)

def calculate_3d_dice(pred_vol, true_vol):
    intersection = np.sum(pred_vol * true_vol)
    volume_sum = np.sum(pred_vol) + np.sum(true_vol)
    if volume_sum == 0:
        return 1.0
    return (2.0 * intersection) / volume_sum

def calculate_clinical_tversky(pred_vol, true_vol, alpha=0.3, beta=0.7):
    p, g = pred_vol.flatten(), true_vol.flatten()
    TP = np.sum(p * g)
    FP = np.sum(p * (1 - g))
    FN = np.sum((1 - p) * g)
    if np.sum(g) == 0:
        return 1.0 if np.sum(p) == 0 else 0.0
    return TP / (TP + (alpha * FP) + (beta * FN))

def calculate_hd95(pred_vol, true_vol):
    if pred_vol.sum() == 0 and true_vol.sum() == 0:
        return 0.0
    if pred_vol.sum() == 0 or true_vol.sum() == 0:
        return np.nan

    pred_bool = pred_vol.astype(bool)
    true_bool = true_vol.astype(bool)

    pred_surface = pred_bool ^ binary_erosion(pred_bool)
    true_surface = true_bool ^ binary_erosion(true_bool)

    dist_pred_to_true = distance_transform_edt(~true_bool)
    dist_true_to_pred = distance_transform_edt(~pred_bool)

    surface_distances_pred = dist_pred_to_true[pred_surface]
    surface_distances_true = dist_true_to_pred[true_surface]

    if len(surface_distances_pred) == 0 or len(surface_distances_true) == 0:
        return np.nan

    all_distances = np.concatenate([surface_distances_pred, surface_distances_true])
    return np.percentile(all_distances, 95) * PIXEL_SPACING_MM

def calculate_surface_dice(pred_vol, true_vol, tolerance_mm=3.0):
    if pred_vol.sum() == 0 and true_vol.sum() == 0:
        return 1.0
    if pred_vol.sum() == 0 or true_vol.sum() == 0:
        return 0.0

    tolerance_px = tolerance_mm / PIXEL_SPACING_MM

    pred_bool = pred_vol.astype(bool)
    true_bool = true_vol.astype(bool)

    pred_surface = pred_bool ^ binary_erosion(pred_bool)
    true_surface = true_bool ^ binary_erosion(true_bool)

    dist_pred_to_true = distance_transform_edt(~true_bool)
    dist_true_to_pred = distance_transform_edt(~pred_bool)

    pred_surface_within_tol = dist_pred_to_true[pred_surface] <= tolerance_px
    true_surface_within_tol = dist_true_to_pred[true_surface] <= tolerance_px

    numerator = pred_surface_within_tol.sum() + true_surface_within_tol.sum()
    denominator = pred_surface.sum() + true_surface.sum()

    return numerator / denominator if denominator > 0 else 1.0

def main():
    print("Loading Swin-UNet weights for full clinical evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = "Swin-UNet"
    model = SwinUNet(in_channels=1, out_channels=2).to(device)
    weights_path = '/home/<hpc-user>/project/checkpoints/swin_unet/best_model.pth'
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded weights from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")

    dataset_path = '/home/<hpc-user>/project/data/dataset.h5'
    split_path   = '/home/<hpc-user>/project/dataset_split.json'

    with open(split_path) as f:
        val_patients = json.load(f)['val']

    # ── OPTIMIZATION 1: The Exception Short-Circuit ──
    print("Pre-filtering to parotid patients only (Optimized & Fixed)...")
    parotid_patients = []
    
    class FoundParotid(Exception): pass

    with h5py.File(dataset_path, 'r') as hf:
        for pid in val_patients:
            if pid not in hf:
                continue
            
            patient_grp = hf[pid]
            has_parotid = False
            
            def check_parotid(name, obj):
                if isinstance(obj, h5py.Group):
                    if 'PAROTID_R' in obj and obj['PAROTID_R'][...].sum() > 0:
                        raise FoundParotid()
                    if 'PAROTID_L' in obj and obj['PAROTID_L'][...].sum() > 0:
                        raise FoundParotid()
            
            try:
                patient_grp.visititems(check_parotid)
            except FoundParotid:
                has_parotid = True
                
            if has_parotid:
                parotid_patients.append(pid)
                
    print(f"Found {len(parotid_patients)} parotid patients out of {len(val_patients)} total")
    val_patients = parotid_patients

    dice_scores    = []
    tversky_scores = []
    hd95_scores    = []
    sdice_scores   = []
    skipped        = 0
    evaluated      = 0

    print(f"\nEvaluating {len(val_patients)} validation patients...\n", flush=True)

    with h5py.File(dataset_path, 'r') as hf:
        for i, patient_id in enumerate(val_patients):
            print(f"\n[{i+1}/{len(val_patients)}] Processing {patient_id}...", flush=True)

            if patient_id not in hf:
                skipped += 1
                continue

            patient_grp = hf[patient_id]
            slice_names = []

            def collect_slices(name, obj):
                if isinstance(obj, h5py.Group) and 'image' in obj:
                    slice_names.append(name)

            patient_grp.visititems(collect_slices)
            slice_names = sorted(slice_names)

            if len(slice_names) == 0:
                skipped += 1
                continue

            vol_pred_r, vol_true_r = [], []
            vol_pred_l, vol_true_l = [], []

            # ── OPTIMIZATION 2: GPU Batch Processing (8 slices at a time) ──
            BATCH_SIZE = 8
            for b_idx in range(0, len(slice_names), BATCH_SIZE):
                batch_names = slice_names[b_idx : b_idx + BATCH_SIZE]
                if (b_idx // BATCH_SIZE) % 2 == 0:
                     print(f"  Processing slices {b_idx+1} to {min(b_idx+BATCH_SIZE, len(slice_names))}/{len(slice_names)}", flush=True)

                batch_tensors = []
                for slice_name in batch_names:
                    grp = patient_grp[slice_name]
                    image_raw = grp['image'][...]
                    image_norm = window_and_normalise(image_raw)
                    batch_tensors.append(torch.tensor(image_norm).unsqueeze(0).unsqueeze(0))
                
                batch_input = torch.cat(batch_tensors, dim=0).to(device)

                with torch.no_grad():
                    output = model(batch_input)
                    pred_prob = torch.sigmoid(output).cpu().numpy()

                for b, slice_name in enumerate(batch_names):
                    grp = patient_grp[slice_name]
                    mask_r = grp['PAROTID_R'][...] if 'PAROTID_R' in grp else np.zeros((512, 512))
                    mask_l = grp['PAROTID_L'][...] if 'PAROTID_L' in grp else np.zeros((512, 512))

                    vol_pred_r.append((pred_prob[b, 0] > 0.5).astype(np.float32))
                    vol_true_r.append(mask_r)
                    vol_pred_l.append((pred_prob[b, 1] > 0.5).astype(np.float32))
                    vol_true_l.append(mask_l)

            vol_pred_r = np.stack(vol_pred_r)
            vol_true_r = np.stack(vol_true_r)
            vol_pred_l = np.stack(vol_pred_l)
            vol_true_l = np.stack(vol_true_l)

            # Skip patients with no parotid annotations
            if vol_true_r.sum() == 0 and vol_true_l.sum() == 0:
                print(f"Patient {patient_id} | No parotid annotations — skipping")
                skipped += 1
                continue

            evaluated += 1

            for p, t in [(vol_pred_r, vol_true_r), (vol_pred_l, vol_true_l)]:
                dice_scores.append(calculate_3d_dice(p, t))
                tversky_scores.append(calculate_clinical_tversky(p, t))
                hd95_scores.append(calculate_hd95(p, t))
                sdice_scores.append(calculate_surface_dice(p, t))

            dice_r = calculate_3d_dice(vol_pred_r, vol_true_r)
            dice_l = calculate_3d_dice(vol_pred_l, vol_true_l)
            print(f"Patient {patient_id} | Dice R: {dice_r:.4f} | Dice L: {dice_l:.4f}", flush=True)

    final_dice    = np.mean(dice_scores)
    final_tversky = np.mean(tversky_scores)
    final_hd95    = np.nanmean(hd95_scores)
    final_sdice   = np.mean(sdice_scores)

    print(f"\nEvaluated: {evaluated} patients | Skipped: {skipped} patients")
    print("\n" + "=" * 50)
    print(f"RESULTS: {model_name}")
    print("=" * 50)
    print(f"Overall 3D Dice        : {final_dice:.4f}")
    print(f"Clinical Tversky       : {final_tversky:.4f}")
    print(f"HD95 (Lower is better) : {final_hd95:.2f} mm")
    print(f"Surface Dice (3mm tol) : {final_sdice:.4f}")
    print("=" * 50)

    results_file = '/home/<hpc-user>/project/results_tracker.csv'
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model_Name', '3D_Dice', 'Tversky', 'HD95_mm', 'Surface_Dice_3mm'])
        writer.writerow([
            model_name,
            f"{final_dice:.4f}",
            f"{final_tversky:.4f}",
            f"{final_hd95:.2f}",
            f"{final_sdice:.4f}"
        ])

    print(f"Results logged to {results_file}")

if __name__ == '__main__':
    main()
