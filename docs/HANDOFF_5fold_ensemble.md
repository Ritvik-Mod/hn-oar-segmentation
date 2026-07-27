# HANDOFF — Parotid nnU-Net: train remaining folds + 5-fold ensemble

You (the receiving agent) previously helped set up a RunPod workflow that trained
a custom nnU-Net parotid model. That model is done and downloaded. **Now the goal
is to train the other 4 folds and ensemble all 5.** Full context below so you can
give exact commands.

## What already exists
- Dataset: **`Dataset002_Parotid`** (nnU-Net raw), 208 train + 43 held-out test.
  L/R-QC-corrected. Labels: `1=parotid_r, 2=parotid_l`. Local copy on the Mac:
  `~/Desktop/Dataset002_Parotid/` (~7 GB, has imagesTr/labelsTr/imagesTs/labelsTs).
- **Fold 0 is trained** with a CUSTOM trainer `nnUNetTrainer_250epochs_noMirror`
  (250 epochs, MIRRORING DISABLED). Config: `3d_fullres`, spacing [3.0,0.977,0.977],
  patch [48,224,192], batch 2. Trained on a RunPod H100 (~1 h), `--npz` used.
- Fold-0 TEST scores (held-out 43 cases): **Dice 0.8187 (R 0.828/L 0.810), HD95
  5.25 mm, Surface-Dice@3mm 0.897**. Local model:
  `~/Desktop/nnunet_results_Dataset002_noMirror/Dataset002_Parotid/nnUNetTrainer_250epochs_noMirror__nnUNetPlans__3d_fullres/`
  (this local copy has fold_0 checkpoints but NOT the validation/ softmax folder —
  it was excluded on download).
- Eval script: `~/Desktop/Projects/ct scan project/pipeline/eval_testset.py`
  (also `.../Parotid_Segmentation_Complete/03_models/nnunet/code/eval_testset.py`).
  Usage: `python eval_testset.py --pred-dir <preds> --gt-dir <labelsTs> --model-name X`.

## GOAL
Train folds 1–4 (same custom trainer, `--npz`), then run the **5-fold ensemble**
prediction on the test set and re-evaluate. Expect ~+1–3 Dice and lower HD95.

## ⚠️ CRITICAL RULES (do not deviate)
1. **DO NOT re-enable mirroring.** Mirroring made L/R parotids interchangeable and
   gave Dice 0.50 / HD95 65 mm (measured). The whole point of the custom trainer is
   that mirroring is OFF. Use `nnUNetTrainer_250epochs_noMirror` for EVERY fold.
2. Every fold must be trained with `--npz` (needed for ensembling/postprocessing).
3. **Budget = $9.** Do NOT use H100 SXM ($3.30/hr → ~$13 for 4 folds). Use a
   cheap GPU: **RTX 4090 or A5000 (~$0.40–0.70/hr)**, ~2–3 h/fold, ~$1–1.5/fold.
   4 folds ≈ $5–6; a full 5-fold-from-scratch ≈ $6–8. Fits with margin.
4. The custom trainer must be REGISTERED in nnU-Net on the pod or training/predict
   fails with `Could not find requested nnunet trainer ...`. See install step.

## STEP 0 — check what survived on the pod / network volume (do FIRST)
SSH in (get fresh IP/port from RunPod "Connect" → "SSH over exposed TCP"), then:
```bash
export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results
ls $nnUNet_raw/Dataset002_Parotid                      # dataset present?
ls $nnUNet_preprocessed/Dataset002_Parotid             # preprocessing cached?
find $nnUNet_results/Dataset002_Parotid -name "checkpoint_final.pth"   # fold_0 there?
ls /workspace/pipeline                                 # scripts + trainer file?
```
- **If `nnUNet_preprocessed/Dataset002_Parotid` exists** → skip upload + preprocessing,
  go to STEP 2 and train folds 1–4 (fold_0 results likely present too → 4 folds only).
- **If empty** (network volume was fresh/lost) → do STEP 1 (reupload) and train ALL
  5 folds (0–4), since the local fold_0 lacks the validation softmax.

## STEP 1 — (only if pod data gone) upload + preprocess
On Mac:
```bash
POD_IP=<from console>; POD_PORT=<from console>
ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519 "mkdir -p /workspace/nnunet/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results} /workspace/pipeline"
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r ~/Desktop/Dataset002_Parotid root@$POD_IP:/workspace/nnunet/nnUNet_raw/
scp -P $POD_PORT -i ~/.ssh/id_ed25519 \
  "~/Desktop/Projects/ct scan project/pipeline/eval_testset.py" \
  root@$POD_IP:/workspace/pipeline/
```
On pod: `nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity`

## STEP 2 — install nnU-Net + REGISTER the custom trainer (every fresh pod)
```bash
pip install -q nnunetv2 nibabel scipy scikit-image
NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cat > $NNUNET_DIR/training/nnUNetTrainer/variants/nnUNetTrainer_250epochs_noMirror.py <<'EOF'
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)


class nnUNetTrainer_250epochs_noMirror(nnUNetTrainerNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
EOF
python3 -c "from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_250epochs_noMirror import nnUNetTrainer_250epochs_noMirror; print('trainer OK')"
```
(NOTE: `__init__` MUST use the explicit signature above, NOT `*args,**kwargs` — the
latter crashes nnU-Net with `KeyError: 'args'`.)

## STEP 3 — train the folds (in tmux, re-export env inside tmux)
For each needed fold (1 2 3 4, plus 0 if starting fresh):
```bash
nnUNetv2_train 2 3d_fullres <FOLD> -tr nnUNetTrainer_250epochs_noMirror --npz
```
Run one at a time in `tmux` (detach Ctrl-b d). ~2–3 h/fold on a 4090. Each prints
`Mean Validation Dice` (~0.79) when done.

## STEP 4 — ensemble + evaluate on the test set
Once all 5 folds are trained:
```bash
# (optional but recommended) pick best config + postprocessing from the 5-fold CV:
nnUNetv2_find_best_configuration 2 -c 3d_fullres -tr nnUNetTrainer_250epochs_noMirror

# 5-fold ENSEMBLE prediction on the held-out test images (averages all folds):
nnUNetv2_predict -i $nnUNet_raw/Dataset002_Parotid/imagesTs \
  -o /workspace/pred_ensemble -d 2 -c 3d_fullres \
  -tr nnUNetTrainer_250epochs_noMirror -f 0 1 2 3 4 --disable_tta

# apply the postprocessing find_best_configuration recommends (it prints the exact
# nnUNetv2_apply_postprocessing command + the pkl/json paths). Then evaluate:
cd /workspace/pipeline
python eval_testset.py --pred-dir /workspace/pred_ensemble \
  --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs --model-name nnUNet_5fold_ensemble
```
(`--disable_tta` is belt-and-suspenders; the noMirror trainer already sets
`inference_allowed_mirroring_axes=None`.)

## STEP 5 — download results (run FIRST, before terminating the pod)
On Mac (skip validation/ to keep it small):
```bash
rsync -avP -e "ssh -p $POD_PORT -i ~/.ssh/id_ed25519" --exclude 'validation/' \
  root@$POD_IP:/workspace/nnunet/nnUNet_results/Dataset002_Parotid \
  ~/Desktop/nnunet_results_Dataset002_5fold
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r \
  root@$POD_IP:/workspace/pred_ensemble ~/Desktop/
```
Then terminate the pod. Report the final ensemble Dice / HD95 / Surface-Dice.

## Cheap wins to also consider (mention to the user)
- **Connected-component post-processing** (largest component per gland) — cheap HD95
  fix, targets stray islands. `find_best_configuration` may enable this automatically.
- The single-side patients (not in Dataset002) could later be added via the masked
  partial-label loss (`pipeline/masked_loss.py`) to grow training data — separate task.
```
