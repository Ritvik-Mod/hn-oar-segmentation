# Run a patient through the model on a GPU pod (inference only)

Conversion (DICOM -> model-ready NIfTI) is done LOCALLY on the Mac (no GPU needed).
The pod only runs the fast GPU inference + rendering. This is inference, so a CHEAP
GPU (RTX 4090 / A5000) is plenty — costs pennies, runs in seconds.

--------------------------------------------------------------------------------
## LOCAL (Mac) — convert (already done for 260594; repeat for any new patient)
```bash
python3 ~/Desktop/Parotid_Segmentation_Complete/05_inference_demo/dicom_to_nifti.py \
  ~/Desktop/<PATIENT_DICOM_FOLDER> ~/Desktop/patient_<ID>_ready <ID>
# -> ~/Desktop/patient_<ID>_ready/imgs/<ID>_0000.nii.gz   (+ gt/<ID>.nii.gz if contoured)
```
For 260594 this already exists at `~/Desktop/patient_260594_ready/`.

--------------------------------------------------------------------------------
## STEP 0 — connect (get IP/port from RunPod "Connect" -> SSH over exposed TCP)
```bash
POD_IP=<new ip>; POD_PORT=<new port>
ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519    # test it opens
```

## STEP 1 — is the model already on the pod? (if you attached the old network volume)
On the pod:
```bash
find /workspace -name "checkpoint_final.pth" -path "*Dataset002*" 2>/dev/null | head
```
- If it lists 5 fold checkpoints -> model is there, SKIP Step 2's model upload.
- If nothing -> do the full Step 2.

## STEP 2 — upload (run on Mac, in a local terminal)
```bash
# make dirs on the pod
ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519 \
  "mkdir -p /workspace/nnunet/nnUNet_results /workspace/patient /workspace/scripts"

# (skip if Step 1 found the model) the 5-fold ensemble model (~470 MB)
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r \
  ~/Desktop/nnunet_results_Dataset002_5fold/Dataset002_Parotid \
  root@$POD_IP:/workspace/nnunet/nnUNet_results/

# the converted patient (image + GT)
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r \
  ~/Desktop/patient_260594_ready/imgs ~/Desktop/patient_260594_ready/gt \
  root@$POD_IP:/workspace/patient/

# the scripts (trainer must be registered; visualizer renders the output)
scp -P $POD_PORT -i ~/.ssh/id_ed25519 \
  ~/Desktop/Parotid_Segmentation_Complete/03_models/nnunet/code/nnUNetTrainer_250epochs_noMirror.py \
  ~/Desktop/Parotid_Segmentation_Complete/05_inference_demo/test_and_visualize.py \
  root@$POD_IP:/workspace/scripts/
```

## STEP 3 — on the pod: install, register trainer, set env
```bash
pip install -q nnunetv2 nibabel scipy scikit-image pydicom matplotlib

NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp /workspace/scripts/nnUNetTrainer_250epochs_noMirror.py $NNUNET_DIR/training/nnUNetTrainer/variants/
python3 -c "from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_250epochs_noMirror import nnUNetTrainer_250epochs_noMirror; print('trainer OK')"

export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed
```

## STEP 4 — predict (5-fold ensemble) + render + score (on the pod)
```bash
# GPU ensemble prediction (seconds per fold on GPU)
nnUNetv2_predict -i /workspace/patient/imgs -o /workspace/patient/preds \
  -d 2 -c 3d_fullres -tr nnUNetTrainer_250epochs_noMirror -f 0 1 2 3 4 --disable_tta

# render 2D montage + 3D + Dice vs the doctor's contour (uses the preds above)
cd /workspace/scripts
python test_and_visualize.py \
  --images /workspace/patient/imgs \
  --pred-dir /workspace/patient/preds \
  --gt-dir /workspace/patient/gt \
  --out /workspace/patient/result --render3d
```
It prints a SUMMARY (mean Dice, HD95). `result/` has `<ID>_slices.png`,
`<ID>_3d.png`, `metrics.csv`.

## STEP 5 — download results (run on Mac) and view the images
```bash
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r \
  root@$POD_IP:/workspace/patient/result ~/Desktop/patient_260594_result
# optional: the raw prediction masks (NIfTI) too
scp -P $POD_PORT -i ~/.ssh/id_ed25519 -r \
  root@$POD_IP:/workspace/patient/preds ~/Desktop/patient_260594_preds

open ~/Desktop/patient_260594_result        # opens the folder; double-click the PNGs
```

## STEP 6 — terminate the pod (stop billing) once the images are downloaded.

--------------------------------------------------------------------------------
### Notes
- No HTML this time — just open the PNGs in `patient_260594_result/`.
- For future patients: run the LOCAL convert step, then upload only the new
  `patient_<ID>_ready/imgs` (+ `gt`), and re-run Steps 4-5 (model already on pod).
- The model is on the pod once uploaded; keep the pod/network volume to skip
  re-uploading it next time.
