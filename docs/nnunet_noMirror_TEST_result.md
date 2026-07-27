# nnU-Net Parotid — No-Mirror retrain: FINAL TEST RESULT (2026-07-07)

Held-out test set (Dataset002_Parotid, 43 cases; corrupt PAR0240 dropped).
Model: `nnUNetTrainer_250epochs_noMirror`, 3d_fullres, fold 0, predicted with `--disable_tta`.

## Test metrics (eval_testset.py)
- 3D Dice           : **0.8187**
- Clinical Tversky  : 0.8166
- HD95 (mm)         : **5.25**
- Surface Dice (3mm): 0.8965
- Per-side          : R Dice 0.8278 | L Dice 0.8095
- nnU-Net internal validation (fold 0): Mean Dice 0.7967

## Before vs after (the story)
| Metric | Mirror-ON (original) | Mirror-OFF (this) |
|---|---|---|
| Internal val Dice | 0.4538 | 0.7967 |
| Test per-side Dice | 0.5030 | 0.8187 |
| HD95 | 65.03 mm | 5.25 mm |
| Surface Dice (3mm) | 0.5816 | 0.8965 |

Diagnosis: nnU-Net's default L/R mirror augmentation (+ test-time mirroring,
`inference_allowed_mirroring_axes (0,1,2)`) made PAROTID_L/PAROTID_R
interchangeable, swapping sides on ~25% of cases (bimodal test distribution,
HD95 ~65 mm). Confirmed via: (a) swap test 11/44, (b) merged L/R-agnostic Dice
already 0.82, (c) dataset L/R relabel fixed only 1/251 (GT was consistent),
(d) disable_tta alone only recovered 0.503 -> 0.564 with HD95 still 64 mm
(so confusion was in the weights, not just TTA). Fix = retrain with
nnUNetTrainerNoMirroring. Result above.

## STATUS
- MODEL RECOVERED & SAVED LOCALLY (2026-07-07). The RunPod pod was only STOPPED
  (not deleted) when balance ran out; /workspace survived the restart and the
  full checkpoint was downloaded.
- Local model dir (usable by nnUNetv2_predict, set nnUNet_results to its parent):
  ~/Desktop/nnunet_results_Dataset002_noMirror/Dataset002_Parotid/
    nnUNetTrainer_250epochs_noMirror__nnUNetPlans__3d_fullres/fold_0/
    checkpoint_final.pth (234 MB) + checkpoint_best.pth + plans/dataset json.
- Test predictions: ~/Desktop/pred_nomirror/ ; CSV: ~/Desktop/results_tracker_TEST.csv
- Reproduce eval locally (CPU): pipeline/eval_testset.py on ~/Desktop/Dataset002_Parotid/labelsTs.

## Local inference (later, no pod needed)
  export nnUNet_results=~/Desktop/nnunet_results_Dataset002_noMirror
  nnUNetv2_predict -i ~/Desktop/Dataset002_Parotid/imagesTs -o ~/Desktop/pred_local \
    -d 2 -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs_noMirror --disable_tta -device cpu
