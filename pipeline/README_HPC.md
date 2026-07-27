# Phase 1 — nnU-Net parotid model on the HPC (rachel)

End-to-end runbook: build the nnU-Net dataset from the per-slice npz, train,
predict on the locked test set, and score it with our metrics.

Everything runs on `rachel` in conda env `deeplearning` (PyTorch 2.4.1 / CUDA 12.1).
Copy the `pipeline/` folder (`build_volumes.py`, `make_nnunet_dataset.py`,
`eval_testset.py`, `nnunet_train.pbs`) to the HPC first, e.g. into
`/home/<hpc-user>/project/pipeline/`.

Paths below assume:
- npz dataset:  `/home/<hpc-user>/project/data/ML_Dataset_Final`
- split file:   `/home/<hpc-user>/project/dataset_split.json`
Adjust if yours differ.

---

## Step 0 — one-time environment setup
```bash
source /apps/anaconda3/bin/activate deeplearning
pip install nnunetv2 nibabel        # torch is already present
```

Set the nnU-Net directories (put these in ~/.bashrc so every job sees them).
NOTE: you already have a `~/project/nnunet` folder from the earlier attempt —
check it first (`ls ~/project/nnunet`) and clear stale contents, or use the
fresh path below:
```bash
export BASE=/home/<hpc-user>/project/nnunet
export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
export nnUNet_results=$BASE/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

## Step 1 — build the nnU-Net dataset from the npz
```bash
cd /home/<hpc-user>/project/pipeline
python make_nnunet_dataset.py \
    --data-dir /home/<hpc-user>/project/data/ML_Dataset_Final \
    --split    /home/<hpc-user>/project/dataset_split.json \
    --out      $nnUNet_raw/Dataset001_Parotid
```
Expected: ~208 training cases (both-parotid patients from train+val) and ~44
test cases written to `imagesTs`/`labelsTs`. Single-side patients are skipped
here on purpose (Phase 2 reclaims them via the masked-loss trainer).

## Step 2 — verify + preprocess (fast, CPU)
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```
This is where nnU-Net auto-configures: it resamples to a common (anisotropy-aware)
spacing, sets the 3D patch size, normalization, and augmentation — the whole
"correct preprocessing" recipe, derived from the data.

## Step 3 — train (GPU, PBS)
```bash
qsub nnunet_train.pbs          # trains 3d_fullres, fold 0, 250-epoch trainer
```
Monitor: `tail -f nnunet_parotid_train.log`. A single fold with the 250-epoch
trainer should finish well inside the 48h walltime on an L40.

Optional stronger model (more time): train all 5 folds (submit fold 0..4 by
editing the last arg), then nnU-Net ensembles them — its built-in answer to the
"multiple experts" idea.

## Step 4 — predict on the locked test set
```bash
nnUNetv2_predict \
    -i $nnUNet_raw/Dataset001_Parotid/imagesTs \
    -o $BASE/test_predictions \
    -d 1 -c 3d_fullres -f 0 -tr nnUNetTrainer_250epochs
```

## Step 5 — score with OUR metrics (comparable to results_tracker.csv)
```bash
python eval_testset.py \
    --pred-dir $BASE/test_predictions \
    --gt-dir   $nnUNet_raw/Dataset001_Parotid/labelsTs \
    --model-name nnUNet_Parotid_3dfullres \
    --results-csv /home/<hpc-user>/project/results_tracker_TEST.csv
```
Prints 3D Dice / Tversky / HD95 / Surface-Dice and appends a row. This is the
**first-ever score on the locked 165-patient test set** (44 of them have both
parotids) — the headline number for the CV.

---

### Notes
- **Target:** the four old models plateaued at ~0.62 3D Dice on *validation*.
  nnU-Net on this data should reach ~0.80+; anything ≥0.75 on the held-out test
  set is a strong, defensible result.
- Full-strength run: drop `-tr nnUNetTrainer_250epochs` everywhere to use the
  1000-epoch default (slower, marginally better).
- Bring the trained checkpoint back to the Mac to swap into the demo (replaces
  the 0.62 Attention U-Net with the new model).
