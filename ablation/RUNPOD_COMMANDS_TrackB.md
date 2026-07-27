# RUNPOD COMMAND SHEET — Track B (P0, A1, E3, E4)

> Copy-paste runbook for the rented pod. Everything here is staged and validated
> locally (CPU/MPS) — no training was run locally. All heavy compute (P0 inference,
> E3/E4 training) is deferred to the GPU per the owner's instruction (local Mac has
> no CUDA; MPS is ~2-3 h for P0 alone, minutes on a GPU).
>
> **Incorporates Track A's hard-won pod learnings** (upload strategy, disk sizing,
> correctness canaries). Read §0 before starting the clock.

---

## 0. PLAN THE UPLOAD BEFORE YOU RENT (this is the money step)

Track B's four tasks have very different data footprints:

| Task | Data needed on pod | Size | Upload cost |
|------|--------------------|------|-------------|
| **P0** | Dataset002 NIfTI (imagesTs+labelsTs) + the 4 `.pth` checkpoints | 6.5 GB + ~2.4 GB ckpts | ~30-40 min |
| **A1** | (reuses P0's saved predictions — no extra data) | — | — |
| **E4** | Dataset002 NIfTI (imagesTr/labelsTr/imagesTs/labelsTs) + plans.json | 6.5 GB (same as P0) | shared with P0 |
| **E3** | a 2D-slice `dataset.h5` for **train+val** patients | **~30 GB** ⚠️ | **~2-3 h / ~$8-10** |

**E3 is the expensive one.** Do NOT naively rsync ML_Dataset_Final (55 GB) or the
HPC dataset.h5 (38 GB). Instead build the compact h5 first (train+val only, parotid
+ image only, test patients dropped):

```bash
# LOCAL, before uploading (CPU/IO; ~30 GB output, faithful reproduction):
python3 ablation_study/E3_transformer_pretrain/build_compact_h5.py \
    --out /path/big/trackB_trainval.h5 --splits train,val
# cost saver (mild caveat, see E3 §): cap negatives, ~7-13 GB:
#   ... --cap-empty-per-patient 40
```

If you would rather not pay the E3 upload, **run P0/A1/E4 first** (cheap Dataset002
upload) and decide on E3 after seeing those results.

### Upload rules (from Track A — these failed silently for them)
- **Never `rsync -H`** on macOS (openrsync defers hardlinks, transfers nothing useful).
- **8 parallel streams**, not one (~420 KB/s single vs ~4.3 MB/s ×8):
  ```bash
  export POD=root@<ip> PORT=<port> KEY=~/.ssh/id_ed25519
  cd "Parotid-Project/Datasets"
  ssh -p $PORT -i $KEY $POD "mkdir -p /workspace/Dataset002_Parotid/{imagesTr,labelsTr,imagesTs,labelsTs}"
  find Dataset002_Parotid -type f | xargs -P 8 -I{} \
      rsync -a --partial -e "ssh -p $PORT -i $KEY" {} $POD:/workspace/{}
  ```
- **Don't tar-gz .nii.gz** (already gzipped; re-compress ratio 1.000).
- **Oversize the volume disk** (Set Volume Disk ~200-400 GB; it's ~free). The default
  50 GB `/workspace` is NOT enough once E4 preprocessing lands (~40 GB for 208 cases).
- After any Stop→Edit→Start to resize, **the SSH PORT changes.**
- Run long jobs in **tmux** (`Ctrl-b d` detach, `tmux attach`).

### Environment
```bash
pip install torch torchvision timm nibabel scipy scikit-image h5py scikit-learn
# torchvision + timm need internet for the E3 pretrained weights (verified they download).
```

---

## 1. P0 — 4 Phase-1 checkpoints on the locked 43-case test set  (fast, ~5-10 min)

Upload `checkpoints/{unet,attention_unet,transunet,swin_unet}/*.pth` and Dataset002.
Then:

```bash
cd ablation_study/P0_phase1_on_test
python3 evaluate_phase1_on_test.py --device cuda --save-preds
#   -> eval.csv (4 rows), per_case.csv (43x4), preds/<model>__PARxxxx.npz
```
- Scores on the **same 43 cases** as nnU-Net (Dataset002 labelsTs); Dice is directly
  comparable to nnU-Net's 0.8187. HD95/Surf-Dice use the Phase-1 **isotropic 0.977**
  convention (caveat vs nnU-Net's anisotropic — reported, see RESULT.md).
- `--save-preds` writes the prediction volumes A1 consumes. **Required for A1.**
- (Fallback: runs on `--device cpu`/`mps` too, just ~2-3 h.)

## 2. A1 — Swin boundary-failure analysis  (CPU, seconds; needs P0 preds)

```bash
cd ablation_study/A1_swin_failure
python3 analyze_swin_failure.py      # per_gland.csv + summary.json (Swin vs U-Net)
python3 make_montage.py              # figures/montage_islands.png + montage_coarse.png
```
Can run on the pod right after P0, or download `P0.../preds/` and run locally (light).

---

## 3. E3 — ImageNet-pretrained TransUNet + Swin-UNet  (2 × ~2D training)

Needs the compact `trainval.h5` (§0) + `dataset_split.json`. Reproduces the Phase-1
2D protocol (Dice+BCE, Adam 1e-4, batch 8, AMP, weighted sampler 20:1, hflip).

```bash
cd ablation_study/E3_transformer_pretrain
DATA=/workspace ; SPLIT=/workspace/dataset_split.json

# TransUNet: clean ImageNet ResNet-50 encoder (258/258 tensors, conv1 3->1 averaged)
python3 train_e3.py --model transunet_pretrained \
    --data $DATA/trainval.h5 --split $SPLIT --save-dir ./ckpt_transunet_pretrained
python3 evaluate_e3.py --model transunet_pretrained \
    --ckpt ./ckpt_transunet_pretrained/best_model.pth --tag TransUNet_pretrained

# Swin: timm ImageNet Swin-T encoder + from-scratch decoder  (⚠️ arch caveat, see RESULT.md)
python3 train_e3.py --model swin_pretrained \
    --data $DATA/trainval.h5 --split $SPLIT --save-dir ./ckpt_swin_pretrained
python3 evaluate_e3.py --model swin_pretrained \
    --ckpt ./ckpt_swin_pretrained/best_model.pth --tag Swin_pretrained
#   -> eval.csv rows, compared against P0's from-scratch TransUNet/Swin
```
Compare each E3 row to the corresponding P0 (from-scratch) row — same 43-case
Phase-1 metric, so pretrained-vs-scratch is apples-to-apples.

## 4. E4 — hand-built 3D U-Net on nnU-Net-preprocessed Dataset002

Needs Dataset002 NIfTI + plans.json (for CTNormalization constants — already in
`Parotid-Project/Results/.../3d_fullres/plans.json`; tiny).

```bash
cd ablation_study/E4_custom_3d_unet
# preprocess (CPU, ~minutes): CTNormalize; spacing already [3,0.977,0.977] so no resample
python3 preprocess_e4.py --out ./preprocessed
# train single fold 0 (nnU-Net default KFold seed 12345 -> 166/42; or pass --splits-json)
python3 train_e4.py --preproc ./preprocessed/train --save-dir ./ckpt_unet3d
# sliding-window inference on the 43 test cases -> nnU-Net-style labelmaps
python3 predict_e4.py --ckpt ./ckpt_unet3d/best_model.pth \
    --preproc ./preprocessed/test --out ./pred_test
# score with the SAME evaluator as nnU-Net (true anisotropic spacing)
python3 ../../pipeline/eval_testset.py --pred-dir ./pred_test \
    --gt-dir "../../Parotid-Project/Datasets/Dataset002_Parotid/labelsTs" \
    --model-name Custom_3D_UNet --results-csv ./eval.csv
```
Compare to nnU-Net 3d_fullres single fold **0.8187** — same data, same preprocessing,
different (plain, vanilla) implementation. Expected ~0.78-0.82.

---

## 5. Correctness canaries (catch silent failures BEFORE trusting numbers)
- **E4 sanity:** `preprocess_e4.py` prints per-case `fg_vox`; the fold split must be
  **166 train / 42 val** (matches nnU-Net's internal fold-0). Assert 43 test predictions.
- **E4 spacing:** eval_testset.py uses true (0.977,0.977,3.0); E4 predictions are saved
  in labelsTs orientation (Y,X,Z) with the labelsTs affine — verified in validate_e4.py.
- **E3:** builder prints `numTraining` and slice counts; the frozen hdf5_dataloader must
  report a sane parotid-slice fraction. The pretrained load prints `258/258` (TransUNet).
- **Never train on test patients.** build_compact_h5.py hard-refuses `test` in --splits.
- Case counts: Dataset002 = 208 train + 43 test everywhere.

## 6. Cost estimate (L40S timings from Track A: 3d_fullres ~37 s/epoch)
- P0+A1: minutes. E4: preprocess ~minutes + train (250 ep × ~40 s ≈ 2.5-3 h) + predict ~10 min.
- E3: 2 × 2D trainings (~2 h each on L40S) + the ~30 GB upload.
- Single L40S is enough; if renting multiple, run E3-transunet / E3-swin / E4 in parallel
  (`CUDA_VISIBLE_DEVICES=N ... &  wait`, one log each).
