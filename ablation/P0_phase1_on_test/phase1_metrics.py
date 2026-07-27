"""
Phase-1 3D volumetric metrics — copied verbatim (behaviourally) from the frozen
`dicom test/<model>_codes_from_hpc/*_evaluate.py` so P0 reproduces exactly the
protocol the four from-scratch models were originally scored with on the val set.

Only change from the originals: PIXEL_SPACING_MM is importable so the caveat
(isotropic 0.977 vs true anisotropic 0.977x0.977x3.0) is explicit and auditable.
Do NOT "improve" these — the point is bit-for-bit comparability with the Phase-1
validation numbers in results_tracker.csv.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

PIXEL_SPACING_MM = 0.977  # Phase-1 isotropic assumption (see master doc 12.3 / 14.5)


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


def window_and_normalise(image):
    """Raw pixel -> HU -> soft-tissue window [-150,250] -> [0,1]. Identical to Phase-1."""
    hu_intercept = -8192.0
    hu_min = -150.0
    hu_max = 250.0
    image_hu = image.astype(np.float32) * 1.0 + hu_intercept
    image_clipped = np.clip(image_hu, hu_min, hu_max)
    return (image_clipped - hu_min) / (hu_max - hu_min)
