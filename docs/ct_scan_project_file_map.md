# CT Scan Project — File System Map
> **Purpose:** This document maps every file and folder inside the `ct scan project` directory so that Claude and Gemini can quickly identify where to access or save any given file. Dataset folders (ML_Dataset_*, raw patient data) are noted but their internals are not listed.

---

## Root: `ct scan project/`

Folders skipped (datasets — do not open/modify unless explicitly told):
- `ML_Dataset_Final/` — Final processed ML dataset (~917 patient entries), the canonical dataset used for training
- `ML_Dataset_Master/` — Master unprocessed dataset (~800 patient entries), the original raw-organized data
- `ML_Dataset_Master_DIRECT_DCM/` — Direct DCM version of master dataset (~110 entries), DICOM files without WC parsing
- `ML_Dataset_Master_DIRECT_DCM_new/` — Small 10-patient DCM subset (used as a new test batch, patient IDs 260251–260275)
- `raw patient data/` — Contains raw DICOM patient folders (dummy and new 10 patients), source data before dataset processing

---

## `ct scan project/Basline UNet Predictions/`
> **Context:** Output folder storing visual prediction results from the baseline UNet model. Each file is a side-by-side or overlay visualization of a CT scan and the model's segmentation prediction. These are the ground-truth comparison plots generated during inference/testing. Do NOT save new model outputs here unless they're from the baseline UNet run.

| File | Info |
|---|---|
| `prediction_test0.png` | Baseline UNet prediction visualization for test sample 0 |
| `prediction_test1.png` | Baseline UNet prediction visualization for test sample 1 |
| `prediction_test2.png` | Baseline UNet prediction visualization for test sample 2 |
| `prediction_test3.png` | Baseline UNet prediction visualization for test sample 3 |
| `prediction_test4.png` | Baseline UNet prediction visualization for test sample 4 |
| `prediction_test5.png` | Baseline UNet prediction visualization for test sample 5 |
| `prediction_test6.png` | Baseline UNet prediction visualization for test sample 6 |
| `prediction_test7.png` | Baseline UNet prediction visualization for test sample 7 |
| `prediction_test8.png` | Baseline UNet prediction visualization for test sample 8 |
| `prediction_test9.png` | Baseline UNet prediction visualization for test sample 9 |

---

## `ct scan project/Pytorch Data Loader Testing/`
> **Context:** Folder used during the PyTorch DataLoader development phase to test and verify that CT scan slices were loading correctly. Contains diagnostic output images (batch visualizations) and two standalone diagnostic plots. The subfolders (`loader testing`, `loader testing1`, `loader testing2`, `loader testing3`) represent successive iterations of testing the dataloader under different configurations — each has 10 batch PNGs except `loader testing2` which has 30.

| File/Folder | Info |
|---|---|
| `hu_diagnostic_fixed.png` | Diagnostic visualization of HU (Hounsfield Unit) value distribution in CT slices, used to verify correct windowing/normalization |
| `ultimate_stat_check.png` | Statistical summary visualization across all slices (mean, std, min, max etc.), used to confirm data integrity |
| `loader testing/` (batch_01–10.png) | First DataLoader test run — 10 batch visualization PNGs; earlier iteration, may have bugs |
| `loader testing1/` (batch_01–10.png) | Second iteration of DataLoader testing — 10 batch PNGs |
| `loader testing2/` (batch_01–30.png) | Third (and most extensive) iteration — 30 batch PNGs; used for thorough sampling validation |
| `loader testing3/` (batch_01–10.png) | Fourth iteration — 10 batch PNGs; likely the final stable loader test output |

---

## `ct scan project/checkpoints/`
> **Context:** Stores saved model weights and training history. This is where trained models should be saved to and loaded from. Currently contains only the UNet checkpoint.

### `ct scan project/checkpoints/unet/`
> **Context:** Checkpoint directory specifically for the Vanilla UNet model. The `best_model.pth` is the best-performing model weights file (saved during HPC training). The `training_history.json` tracks per-epoch metrics. When training new models, save their checkpoints in a similarly named subfolder under `checkpoints/`.

| File | Info |
|---|---|
| `best_model.pth` | PyTorch model weights file (~355 MB) for the best Vanilla UNet checkpoint. Trained on HPC. Use this for inference or as a baseline to compare against. |
| `training_history.json` | Per-epoch training log: train_loss, train_dice, val_loss, val_dice over 14 epochs. Best val_dice ~0.432 at epoch 1. Shows the model was trained for 14 epochs total. |

---

## `ct scan project/day 1 parser building /`
> **Context:** Outputs from the very first day of building the DICOM parser (Day 1 experiments). Contains only two exploratory visualization PNGs — not active code. Mainly kept for historical reference.

| File | Info |
|---|---|
| `dicom_overlay_test.png` | Early test: DICOM slice overlaid with organ mask/contour, used to verify that the parser was reading both image and segmentation data correctly |
| `organ_shapes.png` | Visualization of organ shapes/contours extracted from DICOM files during initial parser exploration |

---

## `ct scan project/dicom test/`
> **Context:** The main working code directory for the entire CT scan / HN segmentation project. Contains all active Python scripts organized into thematic subfolders. This is where you should look for any code-related work and where new scripts should generally be saved.

Also contains:
| File | Info |
|---|---|
| `train_weights_cache.npy` | NumPy array (~610 KB) caching pre-computed per-class training weights used to handle class imbalance in the loss function. Loaded at training time to avoid recomputing. |

---

### `ct scan project/dicom test/Data Handling/`
> **Context:** Scripts for managing, inspecting, and organizing the dataset. These are utility/maintenance scripts, not training scripts. Run these to understand dataset state, count slices, do label standardization, or copy new patient batches into the master dataset.

| File | Info |
|---|---|
| `copy_new10_to_main.py` | Script to copy the new 10-patient batch (from `ML_Dataset_Master_DIRECT_DCM_new`) into the main dataset |
| `documentation.py` | Script that auto-generates or updates project documentation (likely writes stats/summaries to the docs folder) |
| `h&n_count.py` | Counts head-and-neck (H&N) patient entries across the dataset |
| `hn_slice_count.py` | Counts CT slices specifically for HN-positive patients |
| `label_standardisation.py` | Main label standardization script — normalizes inconsistent organ label names across different patient DICOM files and outputs a log |
| `new_10_parotid_count.py` | Counts parotid gland slices in the new 10-patient batch |
| `parotid_count.py` | Counts parotid gland slices across the full dataset |
| `parotid_train_patients.py` | Identifies and saves the list of patients to be used for parotid gland training |
| `splits.py` | Creates train/val/test splits and saves them to `patient classification/dataset_split.json` |
| `verify_dataset.py` | Verifies dataset integrity — checks that all expected files exist and are not corrupted |

---

### `ct scan project/dicom test/Direct DCM/`
> **Context:** Scripts for working directly with raw DICOM files (`.dcm`) without intermediate WC-parsed conversion. Used for the `ML_Dataset_Master_DIRECT_DCM` pathway.

| File | Info |
|---|---|
| `dcm_disk_check.py` | Checks disk space and file counts for the direct DCM dataset |
| `final_direct_.dcm_library_parser.py` | Main parser for the direct DCM approach — reads DICOM files using a library and extracts slices/masks |
| `final_direct_dcm_disk_multi.py` | Multi-process/parallel version of the direct DCM parser — writes processed data to disk at scale |
| `shape_verify_for_dcm_direct.py` | Verifies that the shape/dimensions of processed direct DCM arrays are correct |

---

### `ct scan project/dicom test/Loss Function/`
> **Context:** Contains the active, most up-to-date loss function implementation used for CT segmentation training. This is the **authoritative** version of the loss function for local/Mac development.
>
> ⚠️ **Duplicate note:** `loss_function.py` also exists in `unet_codes_from_hpc/` — that copy is the version that was synced to/from HPC (copied on Mar 18).This version is the most uptodate version. The `Loss Function/` copy (Mar 17) is the local Mac version and may not be uptodate. Always check dates when unsure which is current.

| File | Info |
|---|---|
| `loss_function.py` | Defines the segmentation loss function (likely combined Dice + Cross-Entropy or Tversky loss). Same size as the HPC copy — current as of Mar 17. |
| `loss_function.cpython-313.pyc` | Compiled Python bytecode cache — auto-generated, ignore |

---

### `ct scan project/dicom test/Parser/`
> **Context:** Scripts for parsing WC (Window-Center) formatted DICOM data into numpy arrays — the main data preprocessing pipeline before the direct DCM approach was developed. These produce the `ML_Dataset_Master` style output.

| File | Info |
|---|---|
| `final_wc_parser.py` | Original WC parser — reads DICOM, applies window/center transforms, saves slices as numpy |
| `final_wc_parser_disk.py` | Disk-optimized version of the WC parser — saves directly to disk to avoid memory overflow |
| `final_wc_parser_accu_disk.py` | Accuracy-focused disk version with additional validation checks |
| `final_wc_with_multi.py` | Multi-process/parallel version of the WC parser for speed |
| `shape_verify_for_wc_parsed.py` | Verifies shapes of WC-parsed numpy arrays |
| `tester.py` | Test script for validating parser output |
| `wc_parser_saves_organimage_png.py` | Parser variant that saves organ/mask overlay PNGs alongside numpy arrays (used for visual verification) |

---

### `ct scan project/dicom test/Pytorch DataLoader Stuff/`
> **Context:** Contains the active PyTorch DataLoader and supporting diagnostic scripts for Mac/local development. This is the **authoritative** local version of the dataloader.
>
> ⚠️ **Duplicate note:** `pytorch_dataloader.py` also exists in `unet_codes_from_hpc/` (same size, copied on Mar 18). That copy is the HPC-synced version. The version here (Mar 17) is the local Mac version. The HPC version is the most up-to-date version.

| File | Info |
|---|---|
| `pytorch_dataloader.py` | Main PyTorch Dataset/DataLoader class for loading CT slices and masks during training. Used by `unet_train.py`. |
| `loader_check.py` | Script to sanity-check the DataLoader output — prints shapes, values, batch stats |
| `hu_histogram.py` | Plots a histogram of HU values across a sample of data to verify normalization |
| `hu_verify_gem.py` | HU verification script (may have been written/verified with Gemini's help — "gem" suffix) — checks HU range |
| `ultimate_stat_check.py` | Comprehensive stats check across the full dataset through the DataLoader |
| `pytorch_dataloader.cpython-313.pyc` | Compiled bytecode — auto-generated, ignore |

---

### `ct scan project/dicom test/Unet-Mac/`
> **Context:** The local Mac working version of the UNet model and training script. This is where local (non-HPC) experiments and modifications are made before pushing to HPC. Consider this the **active development version** for local runs.
>
> ⚠️ **Duplicate note:** `unet.py` and `unet_train.py` also exist in `unet_codes_from_hpc/`. `unet.py` is same size (identical). `unet_train.py` is **slightly different** (6429 bytes here vs 6347 in HPC copy) — the local Mac version has local modifications. The HPC version is the most up-to-date version and if dummy testing is to be done then only use mac version if we need stuff for final testing or main work then use files from hpc folder.

| File | Info |
|---|---|
| `unet.py` | UNet model architecture definition in PyTorch (encoder-decoder with skip connections) |
| `unet_train.py` | Training script for the UNet — loads data via pytorch_dataloader, runs training loop, saves best checkpoint. Mac-local version with local modifications vs HPC version. |
| `unet.cpython-313.pyc` | Compiled bytecode — auto-generated, ignore |

---

### `ct scan project/dicom test/attention_unet_codes_from_hpc/`
> **Context:** Scripts copied from HPC for the Attention UNet model — the next model being developed/trained after the baseline UNet. These are the HPC versions intended to be run on the cluster. Do not treat as local development files; sync from HPC when updated.

| File | Info |
|---|---|
| `attention_unet.py` | Attention UNet model architecture — UNet with attention gates for better organ localization |
| `attention_train.py` | Training script for the Attention UNet, HPC version |
| `attention_job.sh` | SLURM/HPC job submission shell script for running Attention UNet training on the cluster |

---

### `ct scan project/dicom test/unet_codes_from_hpc/`
> **Context:** A bundle of scripts copied from HPC representing the full UNet codebase as it exists on the cluster. This is a snapshot (as of Mar 18) of the HPC working directory. Additional file `hdf5_dataloader.py` and `convert_to_h5.py` are present here but NOT in the local Mac folders — these are HPC-specific data pipeline variants.
>
> ⚠️ **Duplicates here vs local folders:**
> - `loss_function.py` — same as `Loss Function/` (Mar 17) but timestamped Mar 18 (HPC copy)
> - `pytorch_dataloader.py` — same as `Pytorch DataLoader Stuff/` (Mar 17) but timestamped Mar 18 (HPC copy)
> - `unet.py` — identical to `Unet-Mac/unet.py`
> - `unet_train.py` — **slightly different** from `Unet-Mac/unet_train.py` (6347 vs 6429 bytes); this is the HPC version

| File | Info |
|---|---|
| `unet.py` | UNet model architecture — HPC copy (identical to Unet-Mac version) |
| `unet_train.py` | Training script — HPC version (82 bytes smaller than Unet-Mac version, slight divergence) |
| `pytorch_dataloader.py` | DataLoader — HPC copy (identical to Pytorch DataLoader Stuff version) |
| `loss_function.py` | Loss function — HPC copy (identical to Loss Function version) |
| `hdf5_dataloader.py` | HPC-only script: DataLoader that reads from HDF5 format (not used locally) |
| `convert_to_h5.py` | HPC-only script: converts the numpy/DICOM dataset to HDF5 format for faster I/O on cluster |
| `check_prediction.py` | Script to load a saved model checkpoint and visualize its predictions on test samples |
| `__pycache__/` | Compiled bytecode cache directory — auto-generated, ignore |

---

## `ct scan project/documentation/`
> **Context:** Project documentation and results tracking. Save any new documentation, reports, or metrics summaries here.

| File | Info |
|---|---|
| `Dataset_Overview.pdf` | PDF overview of the dataset structure, patient counts, label distribution, etc. (~92 KB) |
| `HN_Segmentation_Project_Documentation.docx` | Main Word document documenting the HN (Head & Neck) segmentation project — methodology, pipeline, notes (~27 KB) |
| `results_tracker.csv` | CSV tracking model results. Current entry: Vanilla UNet Baseline — 3D Dice: 0.6209, Tversky: 0.6247, HD95: 4.74 mm, Surface Dice (3mm): 0.7217. Add new model results here as a new row. |

---

## `ct scan project/label standardisation/`
> **Context:** Logs generated by running `label_standardisation.py`. These JSON files record how organ labels were mapped/renamed across patients to ensure consistency. Do NOT manually edit these — they are auto-generated outputs.

| File | Info |
|---|---|
| `label_standardisation_log.json` | Full standardization log for the entire WC-parsed dataset (~17 MB) — maps original label names to standardized names for all patients |
| `label_standardisation_log_dcm.json` | Standardization log for the direct DCM dataset (~2.5 MB) |
| `label_standardisation_log_new10.json` | Standardization log for the new 10-patient batch only (~77 KB) |

---

## `ct scan project/patient classification/`
> **Context:** JSON files that classify patients by category and define dataset splits. These files drive which patients go into train/val/test sets and which patients have specific organs. Load these when setting up the DataLoader or running training.

| File | Info |
|---|---|
| `hn_patients.json` | List of patients classified as HN (Head & Neck) positive — used for HN-specific training (~11 KB) |
| `hn_patients_new10.json` | HN patient list for just the new 10-patient batch (~122 bytes) |
| `non_hn_patients.json` | List of patients NOT classified as HN — for background/control classification (~7 KB) |
| `non_hn_patients_new10.json` | Non-HN list for the new 10-patient batch (~2 bytes, likely empty or 1 entry) |
| `dataset_split.json` | Train/val/test patient split (~13 KB) — the authoritative split file used by the DataLoader. Generated by `Data Handling/splits.py`. |
| `parotid_train_patients.json` | List of patients used specifically for parotid gland training (~4.4 KB) — subset of HN patients |

---

## Key Duplicate File Summary

| File Name | Location A (Local/Mac Dev) | Location B (HPC Copy) | Same? |
|---|---|---|---|
| `loss_function.py` | `dicom test/Loss Function/` (Mar 17) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ✅ Same size (3602 B) |
| `pytorch_dataloader.py` | `dicom test/Pytorch DataLoader Stuff/` (Mar 17) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ✅ Same size (6766 B) |
| `unet.py` | `dicom test/Unet-Mac/` (Mar 15) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ✅ Same size (4662 B) |
| `unet_train.py` | `dicom test/Unet-Mac/` (Mar 17) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ⚠️ DIFFERENT sizes (6429 vs 6347 B) — Mac version has local edits |

**Rule of thumb:**
- For **local Mac development/testing**: use files in `Unet-Mac/`, `Pytorch DataLoader Stuff/`, `Loss Function/`
- For **HPC cluster runs**: use files in `unet_codes_from_hpc/` and `attention_unet_codes_from_hpc/`
- Whenever saving code or file for local mac use create a from_mac dir and do it there if one doesn't exist for the task and whenever doing the same task by saving codes and files from hpc back to mac create a dir from_hpc and save and do stuff there.

---

## Where to Save New Files

| If you're creating... | Save it here |
|---|---|
| New model architecture `.py` | `dicom test/` (in its own subfolder, e.g. `VNet-Mac/`) |
| New training script | `dicom test/<model>-Mac/` for local, `dicom test/<model>_codes_from_hpc/` for HPC version |
| New loss function variant | `dicom test/Loss Function/` |
| New DataLoader variant | `dicom test/Pytorch DataLoader Stuff/` |
| New dataset utility/script | `dicom test/Data Handling/` |
| Model checkpoint (weights `.pth`) | `checkpoints/<model_name>/` |
| Training history log | `checkpoints/<model_name>/training_history.json` |
| Model prediction visualizations | Create a new folder like `<ModelName> Predictions/` at root of `ct scan project/` |
| Documentation / project notes | `documentation/` |
| New model's results metrics | Add a row to `documentation/results_tracker.csv` |
| New patient split / classification JSON | `patient classification/` |
| New label standardization log | `label standardisation/` |
| Parser output test PNGs | `day 1 parser building /` (historical) or a new dedicated folder |
| DataLoader batch test PNGs | `Pytorch Data Loader Testing/` in a new `loader testingN/` subfolder |

## Incase of doubts

- If you are still unsure even 1% unsure about anything after reading this then always ask me before providing any command or making any changes.
- Incase of major code or file downloads or transfer or creation just confirm what you are planning to do with me before providing command or code.