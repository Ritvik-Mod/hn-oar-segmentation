# Cloud retrain runbook — no-mirror model + TTA diagnostic (RunPod H100)

Goal: (Step 1) quantify how much of the L/R flip is test-time-mirroring, then
(Step 2) retrain on the corrected dataset with mirroring OFF for a native clean
per-side result. Both run in ONE pod session (~$5, ~1.5 h).

Placeholders: replace `PORT` and `IP` with your pod's SSH-over-TCP values.

---

## 0. Rent + connect
- RunPod → Deploy → 1× H100 SXM, PyTorch template, SSH key added.
- (Optional: attach a **persistent network volume** so you never re-upload again.)
- Connect: `ssh root@IP -p PORT -i ~/.ssh/id_ed25519`

## 1. Upload from your Mac (new terminal tab)
```bash
# corrected dataset (~7 GB, resumable)
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
  ~/Desktop/Dataset002_Parotid/ \
  root@IP:/workspace/nnunet/nnUNet_raw/Dataset002_Parotid/

# the ALREADY-trained model (for the Step-1 diagnostic)
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
  ~/Desktop/nnunet_results_Dataset001_Parotid/Dataset001_Parotid/ \
  root@IP:/workspace/nnunet/nnUNet_results/Dataset001_Parotid/

# pipeline scripts (eval + the no-mirror trainer)
cd "/Users/ritvikmod/Desktop/Projects/ct scan project"
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" pipeline/ root@IP:/workspace/pipeline/
```

## 2. Pod setup (env + install + register the no-mirror trainer)
```bash
pip install -q nnunetv2 nibabel
export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results

# install the custom trainer into the nnU-Net package
NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp /workspace/pipeline/nnUNetTrainer_250epochs_noMirror.py \
   $NNUNET_DIR/training/nnUNetTrainer/variants/
python3 -c "from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_250epochs_noMirror import nnUNetTrainer_250epochs_noMirror; print('trainer OK')"

tmux new -s work
# (re-export the 3 nnUNet_* vars inside tmux — new shell)
export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results
```

## 3. STEP 1 — free TTA diagnostic (existing model, mirroring off at inference)
```bash
nnUNetv2_predict \
  -i $nnUNet_raw/Dataset002_Parotid/imagesTs \
  -o /workspace/pred_oldmodel_noTTA \
  -d 1 -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs --disable_tta

cd /workspace/pipeline
python eval_testset.py \
  --pred-dir /workspace/pred_oldmodel_noTTA \
  --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs \
  --model-name oldmodel_noTTA
```
Compare the per-side Dice here to the original 0.503. If it jumps a lot, TTA was
the main culprit; if only partly, training-time mirroring matters too (Step 2 fixes both).

## 4. STEP 2 — retrain with mirroring OFF, predict with TTA off
```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror --npz
# ^ ~1 h on the H100. Detach: Ctrl-b then d.  Reattach: tmux attach -t work

nnUNetv2_predict \
  -i $nnUNet_raw/Dataset002_Parotid/imagesTs \
  -o /workspace/pred_nomirror \
  -d 2 -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs_noMirror --disable_tta

python eval_testset.py \
  --pred-dir /workspace/pred_nomirror \
  --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs \
  --model-name nnUNet_Parotid_noMirror_TEST \
  --results-csv /workspace/results_tracker_TEST.csv
```
This per-side Dice (and the HD95) is the headline number.

## 5. Download results, then TERMINATE the pod
```bash
# on your Mac
scp -P PORT -i ~/.ssh/id_ed25519 -r \
  root@IP:/workspace/nnunet/nnUNet_results/Dataset002_Parotid \
  ~/Desktop/nnunet_results_Dataset002_noMirror
scp -P PORT -i ~/.ssh/id_ed25519 -r \
  root@IP:/workspace/pred_nomirror ~/Desktop/pred_nomirror
```
Then Stop/Terminate the pod.
```
