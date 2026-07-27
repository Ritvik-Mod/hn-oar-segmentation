# AI BRAIN — COMPLETE PROJECT CONTEXT
## HN Organ-at-Risk Segmentation Project
> **PURPOSE:** This document is the authoritative single-source-of-truth brain for any AI tool working on this project. It contains everything needed to understand, navigate, and continue the project without any prior context. Updated: April 2026.

---

## SECTION 0: IDENTITY & ROLE CONTEXT

**Project Owner:** Ritvik Mod — 2nd Year B.Tech CSE, Birla Institute of Technology (BIT) Mesra
**Clinical Advisor:** a practising radiation oncologist (name withheld)
**Hospital Partner:** Elekta Monaco Treatment Planning System (single institution)
**Institution:** BIT Mesra
**HPC:** University HPC — <hpc-host>, username: <hpc-user>
**HPC GPUs:** 2x NVIDIA L40 (46 GB VRAM each), PBS Scheduler
**HPC Environment:** Conda: deeplearning (PyTorch 2.4.1 + CUDA 12.1)
**HPC Job Config:** 1 GPU, 4 CPUs, 40 GB RAM, 48-hour walltime

---

## SECTION 1: THE CLINICAL PROBLEM

### What This Project Solves
Between 800–1,000 head and neck (H&N) cancer patients are treated with radiation therapy at the partner hospital annually. Before treatment, a radiation oncologist must manually contour (trace) every critical anatomical structure on every CT slice. This takes ~30 minutes per patient for ~10 organs. Scaled to 1,000 patients/year, this is a massive clinical bottleneck.

### Why Parotid Glands Are The Critical Structure
- Tumour targets receive 60 Gy minimum dose
- Parotid glands sit adjacent to the target and can tolerate only **20–26 Gy maximum**
- Irradiation beyond tolerance → xerostomia (chronic dry mouth) → dental caries, mandibular necrosis, oral ulceration, impaired swallowing/speech
- These complications are largely **avoidable** with accurate parotid contouring
- Treatment technique: Intensity-Modulated Radiation Therapy (IMRT) — routes beams around parotid glands
- **Accurate parotid contouring is a direct patient safety requirement**

### Current Clinical Workflow State
- No auto-contouring software exists at the hospital (fully manual)
- Treatment planning system: Elekta Monaco (exclusively)
- CT scanner manufacturer: Siemens
- 2 radiation oncologists annotate all patients

---

## SECTION 2: DATASET — COMPLETE SPECIFICATION

### Scale & Source
- All data sourced from Elekta Monaco Treatment Planning System at the partner hospital
- Indian-demographic dataset — **significantly underrepresented** in published H&N segmentation literature (most studies use Western datasets)
- Data collection period: less than one full calendar year

### Raw Numbers
| Metric | Value |
|--------|-------|
| Raw patient folders collected | 1,064 |
| Processed via WC parser pipeline | 797 patients |
| Processed via DICOM-RT pipeline | 117 patients |
| Total unique patients after merging | 914 (zero duplicates verified) |
| Confirmed H&N patients | 844 |
| Non-H&N patients excluded | 70 |
| Corrupt patients excluded (mislabeled masks) | 12 |
| Total annotated slices | 126,879 |
| HDF5 database size | 38 GB |
| Original dataset on disk (.npz format) | 55 GB |
| Annotating clinicians | 2 radiation oncologists |

### Data Quality Control
During dataloader development, **12 patients were identified with mislabeled parotid masks** where the annotation covered the entire head region (mask pixel counts of 20,000+ pixels per slice vs normal 500–5,000 pixels). These 12 were excluded from all splits. Exclusion applied at data loading stage — the underlying HDF5 database was NOT altered.

### Train/Val/Test Split
Split locked with random seed = 42 before any training began. Test set stays locked until all architectures complete validation.

| Split | All H&N Patients | Parotid Patients | Purpose |
|-------|-----------------|-----------------|---------|
| Train | 583 (70%) | 383 | Model weight optimisation |
| Val | 84 (10%) | 43 | Early stopping & model selection |
| Test | 165 (20%) | 87 | Final held-out evaluation only |
| **Total** | **832** | **513** | |

Split is **patient-wise** (not slice-wise) — all slices from a patient appear in exactly one split. Prevents data leakage.

### HDF5 Database Structure
Original 55 GB of 126,879 individual .npz files caused severe I/O bottleneck on HPC. Consolidated to single 38 GB HDF5 file:
```
dataset.h5 / [PATIENT_ID] / [CT_GROUP] / Z_[z_value] / {image, PAROTID_R, PAROTID_L, ...}
```
This reduced data loading overhead by ~10x. HDF5 preserves raw uint16 CT values and uint8 binary masks (no preprocessing baked in).

### CT Image Specifications
| Parameter | Value | Notes |
|-----------|-------|-------|
| Image resolution | 512 × 512 pixels | Uniform across all patients |
| Bit depth | 16-bit unsigned integer | Standard CT Hounsfield Unit range |
| Dominant pixel spacing | 0.977 × 0.977 mm | ~90.5% of all scans |
| Pixel spacing range | 0.643–0.977 mm | Varies by patient field of view |
| Slice thickness | 1.0 mm or 3.0 mm | 1.0 mm for H&N; 3.0 mm for larger regions |
| Average slices per patient | 153 | Equals number of .WC files per patient |
| HU conversion formula | pixel × 1.0 + (−8192.0) | RescaleIntercept from DICOM header |

### Parotid Annotations Specifically
- 513 patients annotated with parotid glands (PAROTID_R and PAROTID_L)
- Primary target structures for this project
- Correctly annotated parotids: 500–5,000 pixels per slice
- Mislabeled parotids (excluded): 20,000+ pixels per slice

---

## SECTION 3: TECHNICAL METHODOLOGY — DATA PIPELINE

### 3.1 Two-Pipeline Architecture

**Why two pipelines?** Data came from two different sources requiring different parsing approaches.

| | Pipeline 1: WC Parser | Pipeline 2: DICOM-RT |
|--|----------------------|----------------------|
| Input format | Proprietary Elekta .WC text files | Standard DICOM RTSTRUCT .dcm files |
| Patients processed | 797 | 117 |
| Core library | Custom parser + pydicom | pydicom ContourSequence |
| Coordinate transform | Affine transformation applied manually | Coordinates pre-transformed by Monaco |
| Output | Verified correct, fuzzy Z-matching | Cleanest output, library-handled |

### 3.2 Reverse Engineering the Elekta Monaco .WC Format
**Critical achievement:** No public documentation exists for the .WC format — no whitepapers, no open-source parsers, no GitHub repos. It was decoded from scratch by:
- Identifying ASCII plaintext structure
- Recognising physical millimetre coordinates
- Decoding polygon headers (ROI ID + coordinate count)
- Locating Z-axis encoding in filenames
- Mapping integer IDs to organ names via contournames translation dictionary

### 3.3 The Coordinate Transformation
Converting continuous mm coordinates to discrete pixel indices required a custom affine transformation. The hospital's physical coordinate system and digital array coordinate system differ in Y-axis orientation.

```
Px = (X - Ox) / Sx
Py = (-Y - Oy) / Sy
```

Where:
- X, Y = physical coordinates from .WC file (mm)
- Ox, Oy = real-world position of top-left pixel (DICOM ImagePositionPatient)
- Sx, Sy = physical size of one pixel (DICOM PixelSpacing)
- Y negation = converts physical anterior-posterior orientation to digital top-down orientation

### 3.4 Label Standardisation
Parallel pass across all 126,879 files using Python's ProcessPoolExecutor (8 CPU cores):
- Inconsistent organ name variants mapped to canonical labels (e.g., RT_PAROTID, RIGHT_PAROTID → PAROTID_R)
- Hardware labels and TPS artifacts removed
- Where two source labels mapped to same canonical name within a slice, masks merged via logical OR
- 115,831 of 126,879 files required at least one modification
- Zero errors recorded

### 3.5 Processing Speed
- Initial sequential pipeline: ~340 patients in 9 hours before RAM saturation
- After refactoring to ProcessPoolExecutor (8 cores): remaining 724 patients processed in under 2 hours

---

## SECTION 4: DEEP LEARNING PIPELINE

### 4.1 Dataloader Design (ParotidDataset)
Custom HDF5-based PyTorch Dataset class — implemented from scratch without MONAI, SimpleITK, or external medical imaging frameworks.

**Operations at runtime:**
1. **HU Windowing:** Raw uint16 pixel values → Hounsfield Units via (pixel × 1.0 + (−8192.0)), clipped to soft-tissue window [−150, 250 HU], normalised to [0, 1]
2. **Output format:** image tensor shape (1, 512, 512) float32; mask tensor shape (2, 512, 512) float32 — channel 0 = PAROTID_R, channel 1 = PAROTID_L
3. **Weighted Sampling:** WeightedRandomSampler — weight 20.0 for slices with parotid annotations, weight 1.0 for background-only slices (addresses extreme class imbalance — parotid pixels occupy tiny fraction of 512×512 field)
4. **Augmentation:** Horizontal flip ONLY. Vertical flips and 90° rotations excluded — they destroy spatial priors of axial CT anatomy (left-right symmetry is valid; up-down inversion is not). When horizontal flip applied, PAROTID_R and PAROTID_L channels are SWAPPED to maintain anatomical correctness.
5. **Validation Filtering:** Validation function skips batches where masks.sum() == 0 to prevent empty slices from dominating average validation loss.

### 4.2 Loss Function
```
CombinedLoss = 0.5 × Dice Loss + 0.5 × BCE Loss (with sigmoid activation, NOT softmax)
```
- Left and right parotid treated as **independent binary segmentation problems**
- Dice component handles class imbalance
- BCE provides stable gradients throughout training

### 4.3 Training Configuration (ALL models — identical hyperparameters for fair comparison)
| Parameter | Value |
|-----------|-------|
| Optimiser | Adam, learning rate 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early stopping patience | 10 epochs |
| Batch size | 8 |
| Maximum epochs | 50 |
| Mixed precision | AMP (autocast + GradScaler) for Attention U-Net, TransUNet, Swin-UNet |
| PBS job config | 1 GPU, 4 CPUs, 40 GB RAM, 48-hour walltime |
| Checkpoint saving | Best model (lowest val_loss) + training history saved every epoch |

### 4.4 Evaluation Framework
All architectures evaluated on validation set (84 patients, 43 with parotid annotations) using **3D patient-level volumetric metrics**. Slices stacked into 3D volumes per patient. Predictions thresholded at 0.5 after sigmoid. Metrics computed per-structure (PAROTID_R and PAROTID_L independently) then averaged.

| Metric | Description |
|--------|-------------|
| 3D Volumetric Dice | Standard overlap metric on full 3D patient volumes |
| Clinical Tversky (α=0.3, β=0.7) | Asymmetric — penalises under-segmentation (FN) more than over-segmentation (FP). Reflects clinical priority of not missing tissue. |
| HD95 (mm) | 95th percentile Hausdorff Distance in mm. Worst-case boundary error. PIXEL_SPACING_MM = 0.977 applied for conversion. |
| Surface Dice (3mm tolerance) | Fraction of surface points within 3mm tolerance. Clinically acceptable boundary agreement. |

**Edge case handling:** When both prediction and ground truth are empty for a structure → Dice = 1.0 (correct negative). When only one is empty → HD95 = NaN (excluded via nanmean). Prevents undefined distance calculations from corrupting aggregate metrics.

---

## SECTION 5: ARCHITECTURE IMPLEMENTATIONS

All four architectures implemented **from scratch** in PyTorch without external segmentation frameworks. Each verified against the original published paper before training.

### 5.1 U-Net (Baseline)
- **Paper:** Ronneberger et al. 2015
- **Parameters:** ~31M
- **Architecture:** Standard encoder-decoder with skip connections. 5 levels (64, 128, 256, 512, 1024 channels). MaxPool2d for downsampling, ConvTranspose2d for upsampling. Double 3×3 conv + BatchNorm + ReLU at each level. 1×1 conv output head, no final activation (sigmoid applied in loss).
- **Training:** 14 epochs total, best at epoch 4 (val_loss 0.1685). Early stopping triggered at epoch 14.
- **Best checkpoint:** `checkpoints/unet/best_model.pth` (~355 MB)

### 5.2 Attention U-Net
- **Paper:** Oktay et al. 2018
- **Parameters:** ~31.4M
- **Architecture:** Identical to U-Net baseline but with **attention gates on all 4 skip connections**. Gating signal from decoder (upsampled), skip features from encoder. 1×1 conv projections to intermediate dimension (F_int = F_g/2), element-wise addition, sigmoid attention map, element-wise multiplication with skip features.
- **Training:** 19 epochs total (history recorded from epoch 3 onwards due to mid-training script update to add AMP). Best at epoch 10 (val_loss 0.1637). Early stopping at epoch 19.
- **Note:** Training was interrupted after epoch 2 to add mixed precision (AMP) support and checkpoint resume capability. Model resumed from epoch 2 checkpoint with AMP enabled. This is a compute optimisation only — does not affect learned representations.

### 5.3 TransUNet
- **Paper:** Chen et al. 2021 (arXiv:2102.04306)
- **Parameters:** ~102.5M
- **Architecture:** Hybrid CNN-Transformer. ResNet-50 encoder (first 3 stages with 3, 4, 9 bottleneck blocks) producing 32×32 feature maps at 1024 channels. 1×1 conv patch embedding projecting to 768-dimensional transformer hidden space. 12-layer Vision Transformer encoder with 12 attention heads, MLP dimension 3072, pre-norm architecture (LayerNorm before attention, as in ViT), learned positional embeddings. Cascading Upsampler (CUP) decoder with bilinear 2× upsampling, 3×3 conv + BatchNorm + ReLU blocks, and skip connections from ResNet layer1 (256ch, 128×128) and layer2 (512ch, 64×64). Final segmentation head with 1×1 conv and bilinear upsample to 512×512.
- **Adaptation:** Input modified from 224×224 RGB to 512×512 single-channel CT. Trained from scratch without ImageNet pretraining (controlled comparison). Sequence length = 32×32 = 1024 tokens (manageable for L40 GPU).
- **Training:** 23 epochs total, best at epoch 14 (val_loss 0.1692). Early stopping at epoch 23. Higher training volatility than CNN-based models — consistent with transformer behaviour on medium-sized datasets without pretraining.

### 5.4 Swin-UNet
- **Paper:** Cao et al. 2021 (arXiv:2105.05537, ECCVW 2022)
- **Parameters:** ~27M
- **Architecture:** Pure Transformer U-shaped encoder-decoder with zero convolutional components in the backbone. Patch embedding via 4×4 patches projected to 96-dimensional embeddings. Encoder: 3 stages of Swin Transformer blocks (2 blocks per stage) with Window-based Multi-Head Self-Attention (W-MSA) and Shifted Window attention (SW-MSA) using window size 8. Relative position bias in attention. Patch Merging for 2× downsampling. Decoder mirrors encoder with Patch Expanding for 2× upsampling. Skip connections via concatenation and linear projection. Channel progression: 96, 192, 384, 768 (bottleneck). Final segmentation via linear head at 128×128 resolution with bilinear interpolation to 512×512.
- **Adaptation:** Bilinear interpolation used for final 4× upsample instead of the paper's patch expanding — due to memory constraints on shared GPU (original FinalPatchExpanding creates a 262,144-token tensor that exceeds available VRAM). Trained from scratch without ImageNet-pretrained Swin-T weights.
- **Training:** 22 epochs total, best at epoch 13 (val_loss 0.2101). Early stopping at epoch 22. Significantly higher val_loss than CNN-based models.

### 5.5 nnU-Net v2 (PENDING)
- **Paper:** Isensee et al. 2021, Nature Methods
- **Status:** PENDING. Dataset conversion to NIfTI format complete. Planning and preprocessing complete. Training not yet executed due to HPC GPU contention.
- **Approach:** Self-configuring framework using official nnunetv2 package. Will use automatically determined preprocessing, architecture, training schedule, and augmentation pipeline. NOT a manually implemented architecture — documented separately from the four controlled-comparison architectures.

---

## SECTION 6: VALIDATION RESULTS (4 of 5 Architectures Complete)

All metrics on validation set: 84 patients, 43 with parotid annotations. Test set remains locked.

| Architecture | 3D Dice | Tversky | HD95 (mm) | Surface Dice (3mm) | Parameters | Best Epoch |
|-------------|---------|---------|----------|-------------------|------------|-----------|
| U-Net | 0.6209 | 0.6247 | 4.74 | 0.7217 | ~31M | 4/14 |
| Attention U-Net | 0.6346 | 0.6440 | 6.83 | 0.7296 | ~31.4M | 10/19 |
| TransUNet | 0.6332 | 0.6423 | **3.58** | **0.7379** | ~102.5M | 14/23 |
| Swin-UNet | 0.5106 | 0.5170 | 12.10 | 0.6269 | ~27M | 13/22 |
| nnU-Net v2 | Pending | Pending | Pending | Pending | Auto | Pending |

### Key Observations
1. **CNN/hybrid models cluster together:** U-Net (0.6209), Attention U-Net (0.6346), TransUNet (0.6332) all achieve 3D Dice in the 0.62–0.63 range despite vastly different parameter counts (31M–102.5M). This suggests a **performance ceiling imposed by dataset and annotation quality** rather than architectural capacity.

2. **Pure transformer underperforms without pretraining:** Swin-UNet (0.5106 Dice) significantly underperforms all CNN-containing architectures. Vision Transformers require large-scale pretraining or substantially larger datasets to match CNN performance. The 513 parotid patients in this dataset appear insufficient for a pure transformer trained from scratch.

3. **TransUNet achieves best boundary metrics:** Lowest HD95 (3.58mm) and highest Surface Dice (0.7379). The hybrid CNN-Transformer approach preserves boundary precision better than pure CNN or pure transformer approaches. ResNet-50 encoder provides strong local feature extraction while the 12-layer ViT captures global anatomical context.

4. **Attention U-Net: better overlap, worse boundaries:** Improves Dice and Tversky over U-Net baseline but has higher HD95 (6.83mm vs 4.74mm). Analysis suggests attention gates occasionally produce small isolated prediction islands several mm from the main gland — minimally affects Dice but severely penalises HD95.

5. **Validation loss ceiling:** All CNN/hybrid models converge to approximately 0.16–0.17 validation loss regardless of architectural capacity. Performance bottleneck is **data-related** (annotation inconsistency, 2D slice-level training without 3D context, single HU window) rather than model-related.

---

## SECTION 7: CLINICAL FINDING — ANNOTATION GAP

### Discovery
During qualitative review of model predictions, a **consistent pattern was observed:** models correctly predicted bilateral parotid gland contours in cases where the clinical ground truth contained only unilateral annotation. Initially suspected to be model hallucination — but clinical review by the project's radiation-oncologist advisor confirmed that **model predictions reflected correct anatomy**.

### Clinical Explanation
When a tumour is strictly on one side of the neck, a clinician may choose NOT to contour the contralateral (healthy-side) parotid gland because it is not at risk in that patient's specific treatment plan. This is a **deliberate clinical decision** to save contouring time (~15 minutes per structure) — NOT an annotation error. The resulting ground truth is incomplete for training purposes but clinically appropriate for that patient's treatment.

### Impact on Metrics
- Cases with **completely absent contours:** Dice returns 1.0 (correct negative). HD95 returns NaN (excluded via nanmean). Do not inflate HD95.
- Cases with **partially annotated contours** (clinician started but stopped early): DO penalise stronger models. A model that correctly extends the contour beyond the incomplete annotation receives distance penalties proportional to how much correct anatomy it identifies beyond the ground truth boundary.

### Implications for Interpreting Results
- Reported Dice scores are likely **conservative underestimates** for stronger models (Attention U-Net and TransUNet appear to be identifying more correct anatomy than the ground truth captures)
- This is a known limitation of retrospective clinical datasets documented in medical imaging segmentation literature
- Will be supported by clinician-reviewed case examples in the final paper

---

## SECTION 8: HPC INFRASTRUCTURE AND CHALLENGES

### System Configuration
| Component | Specification |
|-----------|--------------|
| Server | <hpc-host> |
| Username | <hpc-user> |
| GPU Node | rachel-gpu, 2× NVIDIA L40 (46 GB VRAM each) |
| Scheduler | PBS (Portable Batch System) |
| Environment | Conda: deeplearning (PyTorch 2.4.1 + CUDA 12.1) |
| Job Configuration | 1 GPU, 4 CPUs, 40 GB RAM, 48-hour walltime |

### GPU Contention Problem
The rachel-gpu node experienced **persistent GPU contention** from another user's long-running process (robo_env) which occupied 38–45 GB on one or both L40 GPUs continuously for multiple weeks. This caused:
- Repeated out-of-memory (OOM) failures when training jobs were assigned to the occupied GPU
- Workarounds: repeated PBS resubmissions until landing on a free GPU, forcing evaluation scripts to CPU fallback, replanning nnU-Net with an 8 GB GPU memory target

### PBS Job Management Learnings
- PBS does NOT write to -o/-e log files until job completion; all output is spooled on the compute node during execution
- Live monitoring: redirect stdout/stderr within the job script using `exec > logfile 2>&1` combined with Python's -u flag for unbuffered output
- Dataloader worker processes caused **deadlocks** on the HPC shared memory system → fixed by setting nnUNet_n_proc_DA=0 for nnU-Net and using num_workers=4 for custom training scripts

---

## SECTION 9: PROBLEMS ENCOUNTERED AND LESSONS LEARNED

### 9.1 CUDA_VISIBLE_DEVICES Hardcoding
Initial Attention U-Net training script hardcoded `os.environ['CUDA_VISIBLE_DEVICES'] = '1'`, bypassing PBS's GPU assignment. When PBS assigned the job to a node where GPU 1 was occupied, training immediately failed with OOM. **Fix:** removed the hardcoding and let PBS handle GPU assignment.

### 9.2 Non-AMP Training Speed
First Attention U-Net run (without AMP) took ~2–2.5 hours per epoch — only ~19 epochs within 48-hour walltime. Training interrupted after epoch 2, restarted with AMP support + checkpoint resume capability + training history saved every epoch (to survive walltime kills). **All subsequent architectures used AMP from the start.**

### 9.3 Architecture Verification Against Papers
Each architecture verified against original published paper before training:
- **TransUNet:** Initial implementation lacked ResNet-50 CNN encoder, proper 1×1 conv patch embedding, and CUP decoder. Reimplemented from scratch to match Chen et al. 2021 exactly.
- **nnU-Net:** Implemented using official nnunetv2 package (self-configuring framework). Not manually coded.
- **Attention U-Net:** Verified correct against Oktay et al. 2018 — attention gates on skip connections with gating signal from decoder, element-wise addition, sigmoid attention map.

### 9.4 Swin-UNet Memory Issue
Original Swin-UNet FinalPatchExpanding layer upsampled from 128×128 to 512×512 in a single step — tensor shape (B, 262144, 96) ≈ 200 MB per sample in batch. Caused OOM on shared GPU. **Fix:** replaced FinalPatchExpanding with a linear segmentation head at 128×128 resolution followed by bilinear interpolation to 512×512. This is a common practical adaptation documented in several Swin-UNet implementations.

### 9.5 Evaluation Speed on CPU
GPU contention forced evaluation scripts to run on CPU. For TransUNet (102.5M parameters), CPU evaluation of 84 validation patients was very slow. **Optimisation:** pre-filter patient list to only those with parotid annotations (43 of 84), roughly halving evaluation time.

---

## SECTION 10: FILE SYSTEM MAP

### Root: `ct scan project/`

**Datasets (DO NOT open/modify unless explicitly told):**
- `ML_Dataset_Final/` — Final processed ML dataset (~917 patient entries). **CANONICAL dataset used for training.**
- `ML_Dataset_Master/` — Master unprocessed dataset (~800 patient entries). Original raw-organized data.
- `ML_Dataset_Master_DIRECT_DCM/` — Direct DCM version (~110 entries). DICOM files without WC parsing.
- `ML_Dataset_Master_DIRECT_DCM_new/` — Small 10-patient DCM subset (patient IDs 260251–260275).
- `raw patient data/` — Raw DICOM patient folders (dummy and new 10 patients). Source data before processing.

### `checkpoints/`
Where trained model weights are saved and loaded from.
- `unet/best_model.pth` — Best UNet weights (~355 MB). Trained on HPC.
- `unet/training_history.json` — Per-epoch metrics: train_loss, train_dice, val_loss, val_dice over 14 epochs. Best val_dice ~0.432 at epoch 1.
- `attention_unet/` — Attention UNet checkpoint (saved from HPC)
- `swin_unet/` — Swin-UNet checkpoint (saved from HPC)
- `transunet/` — TransUNet checkpoint (saved from HPC)

### `Predictions/`
Visual prediction outputs per model:
- `Basline UNet Predictions/` — prediction_test0–9.png
- `Attention UNet Predictions/`
- `SwinUNet Predictions/`
- `TransUNet Predictions/`

### `documentation/`
- `Dataset_Overview.pdf` — Dataset structure, patient counts, label distribution
- `HN_Segmentation_Project_Documentation_Updated_V2.docx` — Main project documentation (V2, April 2026)
- `HN_Segmentation_Project_Documentation_V1.docx` — V1 documentation (historical)
- `ct_scan_project_file_map.md` — Full file system map
- `results_tracker.csv` — Model results. Current entries: UNet(0.6209/0.6247/4.74/0.7217), AttentionUNet(0.6346/0.6440/6.83/0.7296), TransUNet(0.6332/0.6423/3.58/0.7379), Swin-UNet(0.5106/0.5170/12.10/0.6269)
- `AI_BRAIN_COMPLETE_CONTEXT.md` — THIS document

### `dicom test/` — MAIN CODE DIRECTORY
All active Python scripts organized into thematic subfolders.

#### `Data Handling/`
Utility and maintenance scripts (NOT training scripts):
- `copy_new10_to_main.py` — Copy new 10-patient batch into main dataset
- `documentation.py` — Auto-generates/updates project documentation
- `h&n_count.py` — Counts H&N patient entries
- `hn_slice_count.py` — Counts slices for HN-positive patients
- `label_standardisation.py` — Main label standardization (normalizes organ label names)
- `new_10_parotid_count.py` — Counts parotid slices in new 10-patient batch
- `parotid_count.py` — Counts parotid slices across full dataset
- `parotid_train_patients.py` — Identifies and saves parotid training patient list
- `splits.py` — Creates train/val/test splits, saves to `patient classification/dataset_split.json`
- `verify_dataset.py` — Verifies dataset integrity

#### `Direct DCM/`
Scripts for working directly with raw DICOM files:
- `dcm_disk_check.py` — Checks disk space and file counts for direct DCM dataset
- `final_direct_.dcm_library_parser.py` — Main direct DCM parser
- `final_direct_dcm_disk_multi.py` — Multi-process parallel direct DCM parser
- `shape_verify_for_dcm_direct.py` — Verifies shapes of processed direct DCM arrays

#### `Loss Function/`
**Authoritative local version of loss function.** Same size as HPC copy — current as of Mar 17.
- `loss_function.py` — CombinedLoss (0.5 × Dice + 0.5 × BCE with sigmoid). **Use this for local/Mac development.**

#### `Parser/`
WC format parsing scripts (main data preprocessing pipeline):
- `final_wc_parser.py` — Original WC parser
- `final_wc_parser_disk.py` — Disk-optimized version
- `final_wc_parser_accu_disk.py` — Accuracy-focused disk version with additional validation
- `final_wc_with_multi.py` — Multi-process parallel version
- `shape_verify_for_wc_parsed.py` — Verifies shapes of WC-parsed arrays
- `tester.py` — Test script for validating parser output
- `wc_parser_saves_organimage_png.py` — Parser variant that saves overlay PNGs for visual verification

#### `Pytorch DataLoader Stuff/`
**Authoritative local version of the dataloader.** Same size as HPC copy — local Mac version (Mar 17).
- `pytorch_dataloader.py` — Main PyTorch Dataset/DataLoader class (ParotidDataset)
- `loader_check.py` — Sanity-check DataLoader output
- `hu_histogram.py` — Plots HU value histogram to verify normalization
- `hu_verify_gem.py` — HU range verification script
- `ultimate_stat_check.py` — Comprehensive stats check through the DataLoader

#### `Unet-Mac/`
**Local Mac working version of UNet** — active development before pushing to HPC:
- `unet.py` — UNet architecture (identical to HPC version)
- `unet_train.py` — Training script — ⚠️ DIFFERENT from HPC version (6429 vs 6347 bytes). Mac version has local modifications.

#### `unet_codes_from_hpc/`
Full UNet codebase as it exists on HPC cluster (snapshot as of Mar 18):
- `unet.py` — Architecture (identical to Unet-Mac)
- `unet_train.py` — **HPC version** (82 bytes smaller than Mac version)
- `pytorch_dataloader.py` — DataLoader (identical to Pytorch DataLoader Stuff version)
- `loss_function.py` — Loss function (same as Loss Function/ version)
- `hdf5_dataloader.py` — **HPC-only:** DataLoader reading from HDF5 format
- `convert_to_h5.py` — **HPC-only:** Converts numpy/DICOM dataset to HDF5 for faster cluster I/O
- `check_prediction.py` — Loads saved checkpoint and visualises predictions on test samples

#### `attention_unet_codes_from_hpc/`
Attention UNet codebase from HPC:
- `attention_unet.py` — Attention UNet architecture with attention gates
- `attention_train.py` — Training script (HPC version)
- `attention_job.sh` — PBS job submission shell script

#### `swinunet_codes_from_hpc/`
Swin-UNet codebase from HPC.

#### `transunet_codes_from_hpc/`
TransUNet codebase from HPC.

#### `Trans UNet Mac/`
Local Mac version of TransUNet.

### `patient classification/`
JSON files driving which patients go into which split:
- `hn_patients.json` — All HN-positive patients (~11 KB)
- `hn_patients_new10.json` — HN list for new 10-patient batch (~122 bytes)
- `non_hn_patients.json` — Non-HN patients (~7 KB)
- `non_hn_patients_new10.json` — Non-HN list for new 10-patient batch
- `dataset_split.json` — **Authoritative train/val/test split** (~13 KB). Generated by splits.py. Used by DataLoader.
- `parotid_train_patients.json` — Patients used for parotid gland training (~4.4 KB)

### `label standardisation/`
Auto-generated JSON logs from label_standardisation.py (DO NOT manually edit):
- `label_standardisation_log.json` — Full standardisation log for WC-parsed dataset (~17 MB)
- `label_standardisation_log_dcm.json` — Log for direct DCM dataset (~2.5 MB)
- `label_standardisation_log_new10.json` — Log for new 10-patient batch (~77 KB)

### `Pytorch Data Loader Testing/`
Outputs from PyTorch DataLoader development and verification:
- `hu_diagnostic_fixed.png` — HU value distribution diagnostic
- `ultimate_stat_check.png` — Statistical summary across all slices
- `loader testing/` (batch_01–10.png) — 1st iteration
- `loader testing1/` (batch_01–10.png) — 2nd iteration
- `loader testing2/` (batch_01–30.png) — 3rd iteration (most extensive, 30 batches)
- `loader testing3/` (batch_01–10.png) — 4th iteration (final stable loader test)

### `day 1 parser building /`
Historical reference only:
- `dicom_overlay_test.png` — Early DICOM+mask overlay test
- `organ_shapes.png` — Organ shape extraction from DICOM during initial exploration

---

## SECTION 11: FILE SYSTEM RULES AND CONVENTIONS

### Duplicate File Management
| File Name | Location A (Local/Mac) | Location B (HPC Copy) | Status |
|-----------|----------------------|----------------------|--------|
| `loss_function.py` | `dicom test/Loss Function/` (Mar 17) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ✅ Same size (3602 B) |
| `pytorch_dataloader.py` | `dicom test/Pytorch DataLoader Stuff/` (Mar 17) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ✅ Same size (6766 B) |
| `unet.py` | `dicom test/Unet-Mac/` (Mar 15) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ✅ Same size (4662 B) |
| `unet_train.py` | `dicom test/Unet-Mac/` (Mar 17) | `dicom test/unet_codes_from_hpc/` (Mar 18) | ⚠️ DIFFERENT (6429 vs 6347 B) — Mac version has local edits |

### Rule of Thumb
- **Local Mac development/testing:** use files in `Unet-Mac/`, `Pytorch DataLoader Stuff/`, `Loss Function/`
- **HPC cluster runs:** use files in `unet_codes_from_hpc/` and `attention_unet_codes_from_hpc/`
- **Saving code from Mac:** create a `from_mac` dir if one doesn't exist for the task
- **Saving code from HPC to Mac:** create a `from_hpc` dir

### Where to Save New Files
| Creating... | Save here |
|-------------|-----------|
| New model architecture `.py` | `dicom test/` (own subfolder, e.g. `VNet-Mac/`) |
| New training script | `dicom test/<model>-Mac/` for local, `dicom test/<model>_codes_from_hpc/` for HPC |
| New loss function variant | `dicom test/Loss Function/` |
| New DataLoader variant | `dicom test/Pytorch DataLoader Stuff/` |
| New dataset utility | `dicom test/Data Handling/` |
| Model checkpoint (`.pth`) | `checkpoints/<model_name>/` |
| Training history | `checkpoints/<model_name>/training_history.json` |
| Model prediction visualizations | New folder like `<ModelName> Predictions/` at project root |
| Documentation / notes | `documentation/` |
| New model results | Add row to `documentation/results_tracker.csv` |
| New patient split JSON | `patient classification/` |
| New label standardisation log | `label standardisation/` |

### Critical Rule
**If unsure even 1% about anything — ask before providing any command or making any changes. For major code or file downloads, transfers, or creation — always confirm the plan before executing.**

---

## SECTION 12: CURRENT PROJECT STATUS AND NEXT STEPS

### Completed
- ✅ Full dataset pipeline built and verified (two-pipeline architecture: WC parser + DICOM-RT)
- ✅ Label standardisation across all 126,879 files
- ✅ HDF5 database consolidation (38 GB, hierarchical structure)
- ✅ Custom PyTorch DataLoader (ParotidDataset) with HU windowing, weighted sampling, augmentation
- ✅ Combined loss function (Dice + BCE)
- ✅ U-Net (Baseline) trained and evaluated — 3D Dice: 0.6209
- ✅ Attention U-Net trained and evaluated — 3D Dice: 0.6346
- ✅ TransUNet trained and evaluated — 3D Dice: 0.6332, best boundary metrics
- ✅ Swin-UNet trained and evaluated — 3D Dice: 0.5106
- ✅ Annotation gap discovery — clinical finding documented

### In Progress / Next Steps
- ⏳ nnU-Net v2 training — NIfTI conversion done, planning done, awaiting GPU availability on HPC
- ⏳ Test set evaluation — locked until all 5 architectures complete validation
- ⏳ Final paper write-up — will include clinician-reviewed case examples for annotation gap finding
- ⏳ Potential: 3D training, multi-HU window approach, ImageNet pretraining for transformer models

### Performance Summary (Current Best)
- **Best Dice:** Attention U-Net (0.6346)
- **Best Boundary (HD95):** TransUNet (3.58mm)
- **Best Surface Dice:** TransUNet (0.7379)
- **Worst overall:** Swin-UNet (pure transformer without pretraining)
- **Assessment:** Performance bottleneck is data/annotation quality, not model architecture

---

## SECTION 13: KEY TECHNICAL DECISIONS AND RATIONALE

1. **No MONAI/SimpleITK:** Custom implementations throughout to maintain full control and avoid framework dependencies on HPC.

2. **2D slice-level training (not 3D):** HPC memory constraints. Each 512×512 slice fits comfortably; full 3D volumes would require excessive VRAM.

3. **Sigmoid (not softmax) for output:** Left and right parotids are independent binary problems, not mutually exclusive classes.

4. **Horizontal flip only (not vertical/90° rotations):** Axial CT anatomy has valid left-right symmetry but up-down inversion violates spatial priors. When flipping, mask channels are swapped to maintain L/R anatomical correctness.

5. **WeightedRandomSampler (weight 20x for positive slices):** Extreme class imbalance — parotid pixels are a tiny fraction of the 512×512 field. Without this, model would learn to predict all-background.

6. **Patient-wise split (not slice-wise):** Prevents data leakage where slices from the same patient appear in both train and validation sets.

7. **Locked test set:** Will not be evaluated until all 5 architectures complete validation evaluation. Prevents test set contamination through hyperparameter tuning.

8. **3D Volumetric evaluation (not 2D slice):** More clinically meaningful — aggregates predictions across entire patient volume before computing metrics.

9. **HDF5 consolidation:** 10× improvement in data loading speed on HPC. Essential for practical training with 126,879 slices.

10. **Indian demographic dataset:** A rare contribution — most published H&N datasets use Western populations. This dataset specifically represents an underrepresented population.

---

*This document was auto-generated from the project's source documentation and file system on April 3, 2026. It is the canonical AI brain for this project.*
