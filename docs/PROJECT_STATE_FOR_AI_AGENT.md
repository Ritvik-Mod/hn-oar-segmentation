# HN ORGAN-AT-RISK SEGMENTATION — CURRENT PROJECT STATE (FACTUAL DUMP)

> Purpose: complete, precise description of everything done to date in this project, compiled by directly reading every code file, config, checkpoint history, log, document, and the dataset structure on disk. This is a state report only — it describes what exists, not what to do next. Compiled July 2026 from files last modified March–April 2026.

---

## 1. IDENTITY / ENVIRONMENT

- Owner: Ritvik Mod, 2nd-year B.Tech CSE, BIT Mesra. Clinical advisor: a practising radiation oncologist (name withheld). Single-institution partner using Elekta Monaco TPS and Siemens CT scanners.
- Compute: University HPC `<hpc-host>`, user `<hpc-user>`. Node `rachel-gpu` with 2× NVIDIA L40 (46 GB each). PBS scheduler. Conda env `deeplearning` = PyTorch 2.4.1 + CUDA 12.1. Standard job: 1 GPU, 4 CPU, 40 GB RAM, 48 h walltime.
- HPC project layout referenced in code: `/home/<hpc-user>/project/` with subdirs `code/`, `data/` (`ML_Dataset_Final`, `dataset.h5`), `checkpoints/<model>/`, `logs/`, and `dataset_split.json`.
- Local machine paths in code: `/Users/ritvikmod/Desktop/ct scan project/...` (note: current folder is `/Users/ritvikmod/Desktop/Projects/ct scan project/`; hardcoded paths in scripts point to the older `Desktop/ct scan project/` location and to HPC absolute paths).

## 2. CLINICAL PROBLEM (unchanged across all docs)

800–1,000 H&N cancer patients/year at partner hospital. Manual contouring ≈30 min/patient across ~10 organs. Parotid glands are the primary target structure: tumour needs ≥60 Gy, parotids tolerate only 20–26 Gy; over-dose → xerostomia and downstream complications. No auto-contouring software exists at the hospital currently. Treatment technique is IMRT.

## 3. DATASET — WHAT PHYSICALLY EXISTS ON DISK

Root folder contains these dataset directories (verified counts):
- `ML_Dataset_Final/` — **914 patient directories.** Canonical processed dataset used for training. Each patient: `<patient_id>/<CT_group>/data/*.npz` (nested) or `<patient_id>/data/*.npz` (flat), plus a `verification/` folder of overlay PNGs (`Z_<z>.png`).
- `ML_Dataset_Master/` — 797 entries. Master WC-parsed dataset.
- `ML_Dataset_Master_DIRECT_DCM/` — 107 entries. DICOM-RT parsed, no WC.
- `ML_Dataset_Master_DIRECT_DCM_new/` — 10 entries (patient IDs 260251–260275), the "new 10" batch.
- `raw patient data/dummy patient data/` — raw Elekta source folders (patient IDs 1~200734, 1~240001, 1~240002, 1~240003). Contain `1~CTn/` dirs, `plan/APPROVEDPLAN/` with `.hyp`, `.xlog`, `index.dat`, `contournames`, `info`, `.CT`/`.WC` files.

Total files in the tree ≈ 515,103. Breakdown: ~255,482 PNG (verification overlays), ~255,390 NPZ (per-slice arrays), 1,669 DCM, 1,074 `.WC` + 1,074 `.CT` (raw), plus code/config/log files.

**NPZ contents (verified by loading one file):** each `.npz` holds `image` (512×512, uint16, raw pixel, e.g. min 0 max ~9776) plus one binary uint8 (512×512, values {0,1}) mask array per contoured structure present in that slice. Observed structure keys include: `BODY`, `SPINAL_CORD`, `SPINAL_CORD_PRV`, `PAROTID_L`, `PAROTID_R`, `PTV`, `PTV_MARG`. **So the data contains many organs/structures beyond the parotids**, but only PAROTID_R/PAROTID_L are used by the current training/eval code.

**HDF5 database (`dataset.h5`):** NOT present in this local folder — it lives on HPC at `/home/<hpc-user>/project/data/dataset.h5`. Built by `convert_to_h5.py`. Structure: `dataset.h5/<key>` where key = `<patient>/<CT_group>/data/Z_<z>.npz`, each group holds `image` + `PAROTID_R`/`PAROTID_L` datasets, lzf compression. Reported size 38 GB (vs 55 GB of loose npz).

**CT image specs (from docs):** 512×512, 16-bit unsigned. Dominant pixel spacing 0.977×0.977 mm (~90.5% of scans); full range 0.643–0.977 mm. Slice thickness 1.0 mm (H&N) or 3.0 mm. Avg ~153 slices/patient. HU conversion: `HU = pixel × 1.0 + (−8192.0)`.

### Patient counts and splits (verified from JSON)
- `patient classification/hn_patients.json` → **844** H&N patients. `non_hn_patients.json` → **70**. (844 + 70 = 914 = ML_Dataset_Final dir count.)
- `hn_patients_new10.json` → 10; `non_hn_patients_new10.json` → 0.
- `parotid_train_patients.json` → **383** patients (used as the parotid-positive weighting list).
- `dataset_split.json` → **train 583 / val 84 / test 165** (total **832**). Patient-wise split, seed 42, locked before training. Split keys examples: train `1~240125`; test includes both `1~240003` (WC style) and `260229` (DCM style) IDs.
- Total annotated slices reported: **126,879**. Data-quality exclusion: **12 patients** with mislabeled parotid masks (20,000+ px/slice vs normal 500–5,000) excluded at loading stage; HDF5 not altered.

### ⚠️ Version discrepancies to be aware of (numbers differ between documents)
- AI brain / V2 doc: 844 HN, 513 parotid patients, parotid train list = 383. 
- V1 doc: 904 patients processed, 834 HN, 480 parotid, parotid split 348/43/89, 1,064 raw folders with 267 rejected by QC.
- `results_tracker`/split JSON are the authoritative machine-generated numbers (832 in splits, 844 HN, 383 parotid-train). Treat prose patient counts as approximate/version-dependent.

## 4. DATA PIPELINE (as implemented in code)

Two parsing pipelines feed `ML_Dataset_Final`:
- **WC parser pipeline** (`dicom test/Parser/`): parses proprietary Elekta Monaco `.WC` plaintext contour files (reverse-engineered from scratch — no public spec). Applies affine transform `Px=(X−Ox)/Sx`, `Py=(−Y−Oy)/Sy` using DICOM ImagePositionPatient (Ox,Oy) and PixelSpacing (Sx,Sy). ~797 patients. Scripts: `final_wc_parser.py`, `final_wc_parser_disk.py`, `final_wc_parser_accu_disk.py`, `final_wc_with_multi.py` (ProcessPoolExecutor multiprocess), `wc_parser_saves_organimage_png.py` (writes verification overlays), `shape_verify_for_wc_parsed.py`, `tester.py`.
- **Direct DCM pipeline** (`dicom test/Direct DCM/`): parses standard DICOM RTSTRUCT via pydicom ContourSequence, coordinates pre-transformed by Monaco. ~117 patients. Scripts: `final_direct_.dcm_library_parser.py`, `final_direct_dcm_disk_multi.py`, `dcm_disk_check.py`, `shape_verify_for_dcm_direct.py`.
- **Label standardisation** (`dicom test/Data Handling/label_standardisation.py`): parallel ProcessPoolExecutor pass mapping label-name variants → canonical names, dropping hardware/TPS artifacts, OR-merging duplicate labels. Logs in `label standardisation/`: `label_standardisation_log.json` (102,652 entries), `label_standardisation_log_dcm.json` (13,179), `label_standardisation_log_new10.json` (566). Log values are per-file action lists, e.g. `["RENAMED:patient->BODY","DROPPED:Foam_Core","DROPPED:Carbon_Fiber"]`, `["DROPPED:HEAD_AND_NECK"]`.
- Other `Data Handling/` utilities: `splits.py` (makes dataset_split.json), `parotid_train_patients.py`, `parotid_count.py`, `hn_slice_count.py`, `h&n_count.py`, `verify_dataset.py`, `copy_new10_to_main.py`, `new_10_parotid_count.py`, `documentation.py`.

## 5. DATALOADER / LOSS (verified from code)

**Two dataloaders exist** with a behavioural difference:
- `dicom test/Pytorch DataLoader Stuff/pytorch_dataloader.py` (`ParotidDataset`, reads loose `.npz` via glob). HU window: raw→HU (`×1 −8192`), clip to **[−150, 250]** HU, normalise to [0,1]. Output: image `(1,512,512)` float32, mask `(2,512,512)` float32 (ch0 PAROTID_R, ch1 PAROTID_L). Augment: **horizontal flip only**, and on flip it **swaps R/L mask channels**. Sample weights: **14.0** for slices whose patient is in `parotid_train_patients.json`, else 1.0 (patient-level weighting).
- `dicom test/unet_codes_from_hpc/hdf5_dataloader.py` (`ParotidDataset`, reads `dataset.h5`). Same HU windowing/augment logic, but sample weight = **20.0 for slices that actually contain a parotid mask**, else 1.0 (slice-level weighting). Uses `h5py.visititems`. **This is the version used for all HPC training/eval.**
- ⚠️ The two differ: 14.0 patient-level (local npz) vs 20.0 slice-level (HPC h5). HPC training used the 20.0 slice-level version.
- `convert_to_h5.py` builds the h5 from npz (only image + PAROTID_R/L are packed — other organs are dropped in the h5).

**Loss** (`dicom test/Loss Function/loss_function.py` = `unet_codes_from_hpc/loss_function.py`, identical): `CombinedLoss = 0.5·DiceLoss + 0.5·BCELoss`. Dice computed on `sigmoid(logits)`, per-channel over both batch and spatial dims (one global Dice per L/R channel, then averaged), smooth=1.0. BCE = `nn.BCEWithLogitsLoss`. Returns (total, dice, bce). Sigmoid (independent binary channels), not softmax.

**Validation loop** skips batches where `masks.sum()==0` so empty slices don't dominate val loss.

## 6. MODEL ARCHITECTURES (all hand-implemented in PyTorch, verified in code)

All take `in_channels=1, out_channels=2`, output raw logits (sigmoid in loss).
- **U-Net** (`unet.py`): DoubleConv (Conv-BN-ReLU ×2), Encoder/Decoder, features [64,128,256,512] + 1024 bottleneck, ConvTranspose upsample, 1×1 output head. ~31M params.
- **Attention U-Net** (`attention_unet.py`): U-Net + `AttentionBlock` gates on all 4 skips (F_int=F_g/2, add → sigmoid → multiply). ~31.4M params.
- **TransUNet** (`transunet.py`): `ResNet50Encoder` (stem→128×128; stage1 3 blocks 64→256 @128×128; stage2 4 blocks →512 @64×64; stage3 6 blocks →1024 @32×32), `PatchEmbedding` 1×1 conv →768 hidden, `TransformerBlock` 12 layers/12 heads (MLP 3072, pre-norm, learned pos-embed, 32×32=1024 tokens), CUP decoder with skips from ResNet stage1/stage2, 1×1 seg head + bilinear to 512×512. ~102.5M params. Trained from scratch (no ImageNet), input adapted to 512×512 single-channel. (Note: code comment says stage3 = 6 blocks; AI-brain prose said 9 — the code is authoritative: 3/4/6.)
- **Swin-UNet** (`swin_unet.py`): pure-transformer U-shape. PatchEmbedding 4×4 →96, PatchMerging/PatchExpanding, WindowAttention + shifted-window Swin blocks (2/stage), window size in code, relative position bias, channels 96/192/384/768. **Final upsample replaced** with linear seg head at 128×128 + bilinear→512×512 (memory workaround for original FinalPatchExpanding OOM). ~27M params. Trained from scratch (no ImageNet Swin-T weights).
- **nnU-Net v2**: NOT implemented here as custom code. Planned via official `nnunetv2`. NIfTI conversion + planning/preprocessing reported done; **training NOT executed** (GPU contention). No nnU-Net checkpoint or code in this folder.

Per-model code folders: local (`Unet-Mac/`, `Trans UNet Mac/`) and HPC snapshots (`unet_codes_from_hpc/`, `attention_unet_codes_from_hpc/`, `swinunet_codes_from_hpc/`, `transunet_codes_from_hpc/`). Each HPC folder has `<model>.py`, `<model>_train.py`, `<model>_evaluate.py`/`_predict.py`, and PBS `.sh` job scripts.

## 7. TRAINING CONFIG (identical across the 4 controlled models)

Adam lr=1e-4; ReduceLROnPlateau(patience=5, factor=0.5, mode=min on val_loss); early stopping patience=10; batch size 8; max 50 epochs; AMP (autocast+GradScaler) for Attention/Trans/Swin; checkpoint = best (lowest val_loss) saved as dict `{epoch, model_state_dict, optimizer_state_dict, val_loss}`; history JSON written per run. WeightedRandomSampler(replacement=True) on train.
- ⚠️ `unet_codes_from_hpc/unet_train.py` still contains `os.environ["CUDA_VISIBLE_DEVICES"]="1"` hardcoded at the top (the OOM-causing line the docs say was later removed). Eval/other job scripts instead auto-pick the emptiest GPU via `nvidia-smi ... sort -nr` in the `.sh`.
- PBS job scripts (`.sh`): queue `gpu`, `select=1:ncpus=4:ngpus=1:mem=40gb`, walltime 48:00:00 for training; eval jobs use short walltime and sometimes `workq`/CPU. Use `exec > logfile 2>&1` + `python3 -u` for live logging; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 8. CHECKPOINTS PRESENT ON DISK (`checkpoints/`)

| Model | File | Size | Date |
|---|---|---|---|
| U-Net | `unet/best_model_unet.pth` | 372 MB | Mar 18 |
| Attention U-Net | `attention_unet/best_model_attention.pth` | 377 MB | Mar 24 |
| Attention U-Net | `attention_unet/best_model_final_attention.pth` | 377 MB | Mar 19 |
| TransUNet | `transunet/best_model_transunet.pth` | 1.23 GB | Mar 24 |
| Swin-UNet | `swin_unet/best_model_swinunet.pth` | 336 MB | Mar 24 |

Each model dir also has `training_history_*.json` (per-epoch train_loss/train_dice/val_loss/val_dice). Attention dir additionally has `attention_history.csv` + `attention_learning_curves.png`. No nnU-Net checkpoint. `dicom test/train_weights_cache.npy` caches sample weights.

**Per-epoch history highlights (val_dice here is 2D slice-level, ~0.33; the headline 0.62 figures are separate 3D volumetric eval):**
- U-Net: 14 epochs, best val_loss 0.1685 @ epoch 4 (val_dice≈0.334); early-stopped @14.
- Attention U-Net: history recorded epochs 3–19 (interrupted @ epoch 2 to add AMP+resume), best val_loss 0.1637 @ epoch 10; early-stopped @19.
- TransUNet: 23 epochs, best val_loss 0.1692 @ epoch 14; higher volatility.
- Swin-UNet: 22 epochs, best val_loss 0.2101 @ epoch 13; notably worse than others.

## 9. EVALUATION FRAMEWORK & RESULTS

Evaluation (`*_evaluate.py`) = 3D patient-level volumetric metrics on the **validation** set (84 patients; only those with parotid annotations counted, ~43), predictions thresholded at 0.5 after sigmoid, per-structure (R,L) then averaged. Metrics: 3D Dice; Clinical Tversky (α=0.3, β=0.7, penalises FN); HD95 in mm (`PIXEL_SPACING_MM=0.977`, scipy distance_transform_edt + binary_erosion, `nanmean` aggregation); Surface Dice @3 mm tolerance. Edge cases: both empty→Dice/SurfDice=1.0, HD95=0; one empty→HD95=NaN (excluded). Results appended to `results_tracker.csv`.

**`documentation/results_tracker.csv` (validation set, test set still locked):**

| Model | 3D Dice | Tversky | HD95 (mm) | Surface Dice (3mm) |
|---|---|---|---|---|
| Vanilla_UNet_Baseline | 0.6209 | 0.6247 | 4.74 | 0.7217 |
| Attention_UNet | 0.6346 | 0.6440 | 6.83 | 0.7296 |
| TransUNet | 0.6332 | 0.6423 | 3.58 | 0.7379 |
| Swin-UNet | 0.5106 | 0.5170 | 12.10 | 0.6269 |

Best overlap Dice: Attention U-Net (0.6346). Best boundary (HD95 3.58, SurfDice 0.7379): TransUNet. Worst: Swin-UNet (pure transformer, from scratch). CNN/hybrid models cluster at 0.62–0.63 Dice / ~0.16–0.17 val loss regardless of param count.

**Prediction visualisations** (`Predictions/`): `Basline UNet Predictions/` 10 PNGs, `TransUNet Predictions/` 5, `SwinUNet Predictions/` 2, `Attention UNet Predictions/` 1.

## 10. CLINICAL FINDING ON RECORD (annotation gap)

Models predicted bilateral parotids where ground truth had only unilateral annotation. Dr. Mod confirmed the predictions were anatomically correct — clinicians deliberately omit the contralateral healthy-side parotid when the tumour is one-sided (time-saving, clinically appropriate). Consequence: fully-absent contours don't inflate metrics (Dice=1/HD95=NaN), but partially-annotated cases penalise stronger models, so reported Dice is likely a conservative underestimate for Attention U-Net / TransUNet.

## 11. DOCUMENTATION FILES PRESENT

`documentation/`: `AI_BRAIN_COMPLETE_CONTEXT.md` (authoritative narrative brain, April 2026), `ct_scan_project_file_map.md`, `HN_Segmentation_Project_Documentation_Updated_V2.docx` + `.pdf` + `.pages`, `HN_Segmentation_Project_Documentation_V1.docx`, `Dataset_Overview.pdf`, `results_tracker.csv`. (V1 has extra detail on raw folder structure: CT1 vs CT2 water-phantom, `contournames` dictionary, `T.*.CT`=16-bit DICOM, `T.*.WC`=contour files, `DCMData/`, demographic file withheld for privacy, `plan/` dose data unused.)

## 12. COMPLETED vs OUTSTANDING (state, not recommendations)

**Done:** two-pipeline data extraction; label standardisation (115,831/126,879 files modified, 0 errors); HDF5 consolidation (on HPC); custom dataloader + combined loss; trained + validation-evaluated U-Net, Attention U-Net, TransUNet, Swin-UNet; checkpoints saved; prediction visualisations; annotation-gap clinical finding documented; V1/V2 written docs.

**Outstanding / not yet done:** nnU-Net v2 training (planning done, not trained — no code/checkpoint locally); **test-set evaluation (165 patients) never run — held locked**; no multi-organ model (data has BODY/SPINAL_CORD/PTV/etc. but only parotids modelled); no 3D/2.5D model; no ensembling; no pretraining; final paper not written.

## 13. KNOWN CAVEATS / INCONSISTENCIES (for the agent to keep in mind)

1. Patient-count numbers differ between V1 doc (834 HN / 480 parotid / 904 total) and V2+brain (844 HN / 513 parotid). Machine files: 844 HN, 383 in parotid-train list, splits total 832.
2. Sample-weight value differs: local npz loader = 14.0 (patient-level); HPC h5 loader = 20.0 (slice-level). HPC training used 20.0.
3. `unet_train.py` HPC snapshot still hardcodes `CUDA_VISIBLE_DEVICES="1"`.
4. TransUNet ResNet stage-3 block count is 6 in code (brain prose said 9); code is source of truth.
5. Hardcoded absolute paths in scripts point to `/Users/ritvikmod/Desktop/ct scan project/` (old) and `/home/<hpc-user>/project/` (HPC) — not this folder's current location.
6. `dataset.h5` and NIfTI/nnU-Net artifacts are on HPC, not in this local folder.
7. Reported headline Dice (~0.62) = 3D volumetric val metric; per-epoch history val_dice (~0.33) is a different 2D slice-level quantity — don't conflate them.
8. Two Attention checkpoints exist (`best_model_attention.pth` Mar 24 and `best_model_final_attention.pth` Mar 19).

## 14. DATA LOCATIONS & RUNNABILITY (where things physically live / what can actually run)

**Where the dataset lives:**
- The `.npz` per-slice dataset and the consolidated `dataset.h5` reside on the **HPC** (`/home/<hpc-user>/project/data/`). That is where all training and evaluation actually run.
- The owner **also holds the full dataset on an external hard disk** that can be connected to the Mac on demand. When mounted, the loose-`.npz` loader (`Pytorch DataLoader Stuff/pytorch_dataloader.py`) can read it directly on the Mac; when not mounted, local scripts that expect the data will have nothing to read.
- Therefore the local `ct scan project/` folder should be treated as: **all code + configs + checkpoints + docs always present**, but **bulk image/array data present only when the external drive is connected** (and the HPC copy is the canonical training copy). `dataset.h5` is HPC-only and is not produced locally unless `convert_to_h5.py` is re-run on mounted npz.

**What can run locally on the Mac (no HPC needed):**
- Read/refactor/debug any code; fix paths; write new architectures/dataloaders.
- Inference & prediction visualisation using the on-disk `.pth` checkpoints on a handful of npz slices (CPU or MPS; slow for TransUNet but fine for spot checks). Requires the external drive mounted for real slices.
- Re-run `convert_to_h5.py`, `splits.py`, counting/verification utilities — all require the npz data mounted.

**What requires HPC (cannot be done on the Mac):**
- Full training of any of the 4 models and nnU-Net v2 (needs the L40 GPUs + PBS; TransUNet is ~102.5M params). The Mac is only suitable for small local sanity tests, not full runs.
- Anything touching `dataset.h5`, the NIfTI conversion, or the nnU-Net planning/preprocessing artifacts — those files exist only on HPC.
- Claude Code running on the Mac cannot reach the HPC by itself; HPC jobs are launched via PBS on `rachel`, and results/logs must be pulled back to the Mac (or run Claude Code on the cluster).

**Before running anything, fix stale hardcoded paths:**
- Scripts reference `/Users/ritvikmod/Desktop/ct scan project/` (old — folder is now under `Desktop/Projects/ct scan project/`) and HPC absolute paths `/home/<hpc-user>/project/...`. Update to the current mount point (Mac folder or external-drive path) or HPC path depending on where the job runs.
