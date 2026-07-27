# MASTER PROJECT REFERENCE — Head & Neck Parotid Auto-Segmentation

> **What this document is.** A single, self-contained reference for the entire `ct scan project`
> folder. It is written so that any future AI agent (or human) can answer *any* question about
> this project — a past coding choice, an old result, a design decision, what a specific script
> does, why something was done a particular way — **without having to re-read the whole folder**.
> If you are an agent that has just been handed this project, read this file first; it is the
> authoritative index and explanation of everything.
>
> **Scope note (important).** The owner built several **web-app / showcase artifacts** (a FastAPI
> `webapp/`, a Gradio `demo/`, and several static "showcase" websites under `Parotid-Project/Website/`
> and `Parotid-Project/Visualizations/`). These were throwaway presentation experiments and are
> **deliberately NOT documented in detail here** — the owner asked that they be dropped from working
> context. They are listed once in the file-map (Section 16) only so their existence is not a mystery;
> they are not part of the core research/engineering pipeline and should be ignored unless the owner
> revisits them.
>
> **Authoritative-numbers rule.** Where prose in older docs disagrees with machine-generated files
> (split JSONs, `results_tracker.csv`, code), **the machine files and the code win.** Section 17
> lists every known discrepancy explicitly.
>
> Compiled by reading every code file, config, log, checkpoint history, results CSV, and document
> in the folder. Last compiled: 2026-07-16.

---

## TABLE OF CONTENTS

1. Identity, people, compute environment
2. The clinical problem (why this project exists)
3. Dataset — complete specification
4. Raw hospital data formats (Elekta Monaco `.WC` and DICOM-RTSTRUCT)
5. Parsers — the two extraction pipelines (every script)
6. Label standardisation (every script + logs)
7. Patient classification & the locked split (every JSON)
8. Data-handling utilities (every counting/verification script)
9. Intensity / HU handling and the PyTorch dataloaders
10. Loss functions (combined Dice+BCE, and the masked partial-label loss)
11. Phase 1 — the four from-scratch architectures (every model file)
12. Phase 1 — training config, scripts, PBS jobs, evaluation framework
13. Phase 1 — results (validation 3D metrics + per-epoch histories)
14. Phase 2 — nnU-Net v2: the full 3D pipeline (every script)
15. Phase 2 — the L/R mirror bug: the full diagnostic story + final results
16. Clinical finding (the annotation gap) + the multi-organ data audit/plan
17. Checkpoints, caveats, discrepancies, and what-runs-where
18. Complete file-system map (every folder)
19. Glossary of project-specific terms
20. Ablation study — living log (the July-2026 controlled study; headline correction lives here)
21. Next experiment — masked partial-label loss (planned, number-backed)
22. Independent verification & artifact locations (main-agent audit, 2026-07-17)

---

## 1. IDENTITY, PEOPLE, COMPUTE ENVIRONMENT

- **Owner / author:** Ritvik Mod — 2nd-year B.Tech CSE, Birla Institute of Technology (BIT) Mesra.
  Solo project (~1 month of focused work at the nnU-Net stage, on top of earlier from-scratch work).
- **Clinical advisor:** a practising radiation oncologist (name withheld).
  Provides the ground-truth clinical judgement (e.g. confirmed the annotation-gap finding).
- **Hospital partner:** single institution using the **Elekta Monaco** treatment planning system (TPS),
  **Siemens** CT scanners, **2 radiation oncologists** doing all manual contouring. IMRT technique.
- **Dataset demographic:** Indian population — deliberately noted as *underrepresented* in published
  H&N segmentation literature (most public datasets are Western).

**University HPC (used for Phase 1 + all early work):**
- Server `<hpc-host>`, user `<hpc-user>`.
- Node `rachel-gpu`: **2× NVIDIA L40 (46 GB VRAM each)**. **PBS** scheduler.
- Conda env `deeplearning` = **PyTorch 2.4.1 + CUDA 12.1**.
- Standard job: **1 GPU, 4 CPU, 40 GB RAM, 48 h walltime**, queue `gpu` (eval jobs use `workq`, CPU).
- HPC project layout referenced in code: `/home/<hpc-user>/project/` with `code/`, `data/`
  (`ML_Dataset_Final`, `dataset.h5`), `checkpoints/<model>/`, `logs/`, and `dataset_split.json`.
- **Persistent problem:** GPU contention. Another user's `robo_env` process occupied 38–45 GB on the
  L40s for weeks, causing repeated OOM. This is *why nnU-Net was never trained on HPC* and the owner
  moved to rented cloud GPUs.

**Rented cloud (used for Phase 2 / nnU-Net):**
- **RunPod.** First an **H100 SXM (80 GB)**, PyTorch 2.8.0 template, ~$3.30/hr (single-fold training).
- Then a **4× RTX 4090 / A40** box to train folds 1–4 in parallel for the 5-fold ensemble (~$6, ~2 h).
- Institutional firewall blocked SSH from the cloud IP to HPC, so the ~7 GB dataset was relayed
  **HPC → owner's Mac → pod**. Data on a RunPod **network volume** (survives pod stop/restart).

**Local machine:** the owner's Mac. Current folder location:
`/Users/ritvikmod/Desktop/Academics/Projects/ct scan project/`.
⚠️ Many older scripts hardcode **stale paths** — `/Users/ritvikmod/Desktop/ct scan project/` (old) and
`/Users/ritvikmod/Desktop/Projects/ct scan project/` and HPC absolutes `/home/<hpc-user>/project/`.
Update paths before running anything. The bulk image data is **not always local** — it lives on HPC and
on an **external hard drive** (`/Volumes/Ritvik Mod/AARUNI` was the raw source); the local folder is
reliably only code + configs + checkpoints + docs.

---

## 2. THE CLINICAL PROBLEM (WHY THIS PROJECT EXISTS)

- The partner hospital treats **~800–1,000 head & neck (H&N) cancer patients/year** with radiotherapy.
- Before treatment, an oncologist must **manually contour** every organ-at-risk (OAR) on every CT slice:
  **~30 minutes per patient across ~10 organs**. At 1,000 patients/year this is a major bottleneck.
- **Parotid glands** (the large salivary glands) are the *primary* target structure of this project:
  - The tumour target needs **≥60 Gy**; the parotids tolerate only **20–26 Gy max**.
  - Over-dosing them → **xerostomia** (permanent dry mouth) → dental caries, mandibular necrosis, oral
    ulceration, impaired swallowing/speech — largely **avoidable** with accurate contouring.
  - IMRT (Intensity-Modulated Radiation Therapy) routes beams around the parotids, so accurate parotid
    contours are a **direct patient-safety requirement**.
- The hospital has **no auto-contouring software** (fully manual). This project is **decision-support**
  (assists/accelerates contouring), *not* an autonomous or regulatory-cleared diagnostic tool.
- **Original goal:** a research paper. **Pivoted (July 2026)** to a **portfolio-grade working model**
  for internship applications (deliverables: strong test metrics, clean reproducible repo, a demo).

---

## 3. DATASET — COMPLETE SPECIFICATION

### 3.1 Source & scale
- All data exported from the hospital's **Elekta Monaco TPS**. Collected over **< 1 calendar year**.
- Two export formats → two parsing pipelines (see Sections 4–5).

### 3.2 Headline numbers (from the AI-brain / V2 doc; treat prose counts as approximate)
| Metric | Value |
|--------|-------|
| Raw patient folders collected | 1,064 |
| Processed via WC parser pipeline | ~797 patients |
| Processed via DICOM-RT pipeline | ~117 patients |
| Total unique patients after merge | **914** (zero duplicates verified) |
| Confirmed H&N patients | **844** |
| Non-H&N excluded | 70 |
| Corrupt (mislabeled masks) excluded | 12 |
| Total annotated slices | **126,879** |
| On-disk `.npz` dataset size | ~55 GB |
| Consolidated HDF5 size (HPC) | ~38 GB |
| Annotating clinicians | 2 |

### 3.3 What physically exists on disk (verified)
- `ML_Dataset_Final/` — **914 patient directories.** THE canonical processed dataset used for training.
  Layout per patient is either **nested** `<patient>/<CT_group>/data/Z_<z>.npz` or **flat**
  `<patient>/data/Z_<z>.npz`, plus a `verification/` folder of overlay PNGs (`Z_<z>.png`).
- `ML_Dataset_Master/` — 797 entries. The WC-parsed master (pre-final-organization).
- `ML_Dataset_Master_DIRECT_DCM/` — 107 entries. DICOM-RT parsed, no WC.
- `ML_Dataset_Master_DIRECT_DCM_new/` — 10 entries (patient IDs 260251–260275), the "new 10" batch.
- `raw patient data/` — raw Elekta source folders (dummy set: `1~200734`, `1~240001/2/3`; plus a
  "new 10 patient data" set). Contain `1~CTn/` dirs and `plan/APPROVEDPLAN/` with `.hyp`, `.xlog`,
  `index.dat`, `contournames`, `info`, `.CT`/`.WC` files.

### 3.4 What a `.npz` slice file contains (verified by loading)
Each `Z_<z>.npz` holds:
- `image` — **512×512, uint16, RAW pixel values** (not calibrated HU). Example min 0, max ~9776.
- one **binary uint8 (512×512, values {0,1})** array **per contoured structure present on that slice**.
- Observed keys include: `BODY`, `SPINAL_CORD`, `SPINAL_CORD_PRV`, `PAROTID_L`, `PAROTID_R`, `PTV`,
  `PTV_MARG`, and many more (eyes, optic nerves, larynx, lens, temporal lobe, etc.). **The data holds
  many organs beyond parotids** — but only `PAROTID_R`/`PAROTID_L` are used by the current models.
- ⚠️ Some npz mask keys still contain **non-standardised raw names** (`dx22`, `DLX1`, `X2`, `spill`, …) —
  label standardisation logged actions but did not fully rewrite every key. A cleaning pass is required
  before any multi-organ training.

### 3.5 CT image specifications
| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 512×512 | uniform across all patients |
| Bit depth | 16-bit unsigned | standard CT HU range |
| Dominant in-plane spacing | **0.977 × 0.977 mm** | ~90.5% of scans |
| Spacing range | 0.643–0.977 mm | varies by field of view |
| Slice thickness | 1.0 mm (H&N) or 3.0 mm | |
| Avg slices/patient | ~153 | = number of `.WC` files per patient |
| **HU conversion** | `HU = pixel × 1.0 + (−8192.0)` | RescaleIntercept from DICOM header |

⚠️ **Key intensity fact:** the npz `image` is **raw pixel** (soft-tissue foreground ≈ 8198, i.e. HU+8192).
Models were trained on this space. The dataloader windows raw→HU→clip→normalise at runtime (Section 9).
The npz files do **not** store pixel spacing; **Z is recoverable from the filename**, in-plane X/Y is not,
so 0.977 mm is used as a documented assumption when reconstructing 3D volumes.

### 3.6 The 12 excluded corrupt patients
During dataloader development, 12 patients were found with **mislabeled parotid masks** covering the whole
head region (**20,000+ px/slice** vs a normal parotid's **500–5,000 px/slice**). They were **excluded at
the data-loading stage** — the underlying HDF5 database was **not** altered. (In Phase 2, an automated QC
filter — drop any gland > 3× the dataset median volume — formalises this; see Section 14.)

---

## 4. RAW HOSPITAL DATA FORMATS

The raw source lived on an external drive `/Volumes/Ritvik Mod/AARUNI`, organised as
`<MONTH>/<PATIENT>/…` (month folders like MAY2024, SEP2024, MAY2025). Two contour formats existed:

### 4.1 Elekta Monaco `.WC` format (the hard one — reverse-engineered from scratch)
**No public documentation exists** for the `.WC` format — no whitepapers, no open-source parsers, no
GitHub repos. The owner decoded it from scratch:
- It is **ASCII plaintext**. Each patient's contours are split across many `T.*.WC` files, **one per CT
  slice** (~153 per patient). The Z position is encoded in the **filename** (`extract_z_from_name`
  strips a leading `T.` and regex-parses the number).
- Inside a `.WC` file (parsing starts at line index 7): repeating blocks of
  `[n_points] \n [organ_id] \n [x y x y x y …]` — an integer point count, an integer ROI/organ id, then
  the polygon's physical-millimetre coordinate pairs (a line "has coordinates" if it contains a `.`).
- Integer `organ_id`s are mapped to organ **names** via a per-patient **`contournames`** dictionary file
  (`read_contournames`: name line followed by an `id,…` line).
- Physical spatial metadata (origin, spacing) is borrowed from **any DICOM** in the patient folder
  (`.DCM` or `.CT`) via pydicom (`ImagePositionPatient`, `PixelSpacing`).

**The coordinate transform** (continuous mm → discrete pixel indices), applied per polygon:
```
Px = (X  - Ox) / Sx
Py = (-Y - Oy) / Sy
```
- `X, Y` = physical mm coordinates from the `.WC` file.
- `Ox, Oy` = real-world position of the top-left pixel (DICOM `ImagePositionPatient`).
- `Sx, Sy` = physical size of one pixel (DICOM `PixelSpacing`).
- **The Y negation** converts physical anterior–posterior orientation to digital top-down array
  orientation (the hospital's physical Y-axis and the image array Y-axis are flipped). Polygons are then
  rasterised into binary masks with OpenCV (`cv2.fillPoly`).

### 4.2 DICOM-RTSTRUCT format (the easy one)
Standard DICOM. A single `RTSTRUCT` `.dcm` file holds all contours; the CT is a stack of CT `.dcm`
slices. Parsed with **pydicom**:
- `StructureSetROISequence` → `{ROINumber: ROIName}` map.
- `ROIContourSequence` → per-ROI `ContourSequence`; each contour's `ContourData` is a flat list reshaped
  to `(-1, 3)` (x, y, z in mm). Coordinates are **already transformed by Monaco**, so no manual affine is
  needed — just match each contour's Z to the CT slice's `ImagePositionPatient[2]` and rasterise.
- Includes a **fallback** for CT slices with missing/implicit transfer syntax (injects
  `ImplicitVRLittleEndian`) and for missing pixel data (reads the trailing `rows*cols*2` bytes as int16).
- **IDs:** WC-style patient IDs contain a `~` (e.g. `1~240125`); DICOM-style IDs are numeric (e.g. `260229`).

### 4.3 Other raw files (from the V1 doc, for completeness)
Inside a raw patient folder: `1~CT1` (real CT) vs `1~CT2`/`1~ref1` (reference/water-phantom, usually
skipped — the WC parser strictly targets CT1); `plan/APPROVEDPLAN/` holds dose/plan data (`.hyp`, `.xlog`,
`index.dat`, `info`) which is **unused** by this project; `DCMData/`; `contournames` (the ID→organ dict);
`T.*.CT` = 16-bit DICOM images, `T.*.WC` = contour files. A demographic/patient-info file exists but is
withheld for privacy.

---

## 5. PARSERS — THE TWO EXTRACTION PIPELINES

All parser scripts live in `dicom test/Parser/` (WC) and `dicom test/Direct DCM/` (DICOM-RT), with an
identical mirrored copy under `Parotid-Project/Segmentation_Complete/…/01_parsers/`. They write per-slice
`.npz` (+ verification PNGs) into `ML_Dataset_Master` (WC) or `ML_Dataset_Master_DIRECT_DCM` (DCM).

### 5.1 WC parser family (`dicom test/Parser/`)
- **`final_wc_parser.py`** — the original single-process WC parser. Walks month→patient folders on the
  external drive, targets CT1 only, borrows DICOM metadata, parses `.WC` polygons, applies the affine,
  rasterises masks, saves npz + verification overlays.
- **`final_wc_parser_disk.py`** — disk-optimised variant (writes straight to disk to avoid RAM blow-up).
- **`final_wc_parser_accu_disk.py`** — accuracy-focused disk variant with extra validation.
- **`final_wc_with_multi.py`** — **multiprocess** version (Python `ProcessPoolExecutor`, ~8 cores). This
  is what took the pipeline from "~340 patients in 9 h before RAM saturation" to "remaining 724 in < 2 h".
- **`wc_parser_saves_organimage_png.py`** — variant that additionally writes per-organ overlay PNGs.
- **`shape_verify_for_wc_parsed.py`** — verifies array shapes/dtypes of WC-parsed npz output.
- **`tester.py`** — ad-hoc parser test/validation script.

Uses **fuzzy Z-matching** (WC filename Z vs DICOM slice Z within tolerance) because the two sources'
Z encodings don't always align to the decimal.

### 5.2 Direct-DCM parser family (`dicom test/Direct DCM/`)
- **`final_direct_.dcm_library_parser.py`** — main single-patient DICOM-RTSTRUCT parser (the RAJUBHAI_RATHOD
  example is hardcoded; generalised in the multi version). Cleanest output — pydicom handles the geometry.
- **`final_direct_dcm_disk_multi.py`** — multiprocess batch version over many DICOM patients.
- **`dcm_disk_check.py`** — disk-space / file-count sanity check for the direct-DCM dataset.
- **`shape_verify_for_dcm_direct.py`** — shape verification for direct-DCM npz output.

**Why two pipelines at all:** the data arrived in two formats. WC needed the full reverse-engineered
affine; DICOM-RT was library-handled. Both converge on the same per-slice npz schema so downstream code
is format-agnostic.

---

## 6. LABEL STANDARDISATION

Script: `dicom test/Data Handling/label_standardisation.py` (mirror in `…/02_preprocessing/data_handling/`).
- A **parallel `ProcessPoolExecutor` (8 cores)** pass over all 126,879 npz files.
- **`LABEL_MAP`** normalises inconsistent organ-name variants to canonical names, e.g.
  `RT_PAROTID / RIGHT_PAROTID / RT PAROTID → PAROTID_R`; `LT_PAROTID / LEFT_PAROTID → PAROTID_L`;
  `RT_EYE → EYE_R`; `RT_OPTIC_NERVE → OPTIC_NERVE_R`; `SPINALCORD / CORD → SPINAL_CORD`;
  `BRAIN_STEM → BRAINSTEM`; `patient/PATIENT → BODY`; `prvspine/PRVSPINE/SPINE_PRV → SPINAL_CORD_PRV`;
  `RT_LENS → LENS_R`; etc.
- **`DROP_LABELS`** removes hardware/TPS artifacts and junk: `Foam_Core`, `Carbon_Fiber`, `General`,
  `sampleElekta`, `HEAD_AND_NECK`, `1`, `XX1`, `XX3`, `50`, `DARS`, `DX11`, `PTVMAR1`, `BolusFourPoints`,
  `X1`, `x11`, `P1`, etc.
- When **two source labels map to the same canonical name** within a slice, masks are **merged via logical
  OR** (`np.logical_or`).
- Per-file it records an action log like `["RENAMED:patient->BODY","DROPPED:Foam_Core"]` and only re-saves
  (`np.savez_compressed`) files that changed.
- **Result:** 115,831 of 126,879 files required ≥1 modification; **0 errors**.

**Logs (auto-generated, do NOT hand-edit)** in `label standardisation/` (mirror in `…/label_logs/`):
- `label_standardisation_log.json` — WC dataset, ~102,652 entries (~17 MB).
- `label_standardisation_log_dcm.json` — direct-DCM dataset, ~13,179 entries (~2.5 MB).
- `label_standardisation_log_new10.json` — the new-10 batch, 566 entries (~77 KB).

⚠️ As noted in §3.4, this pass did **not** catch every raw key variant (`dx22`, `X2`, `spill`, …) — a
further canonicalisation pass is needed before multi-organ training.

---

## 7. PATIENT CLASSIFICATION & THE LOCKED SPLIT

Directory `patient classification/` (mirror in `…/02_preprocessing/splits/`). All JSONs.

- **`hn_patients.json`** → **844** H&N-positive patients. **`non_hn_patients.json`** → **70**.
  (844 + 70 = 914 = `ML_Dataset_Final` directory count.) A patient is "H&N" if it contains ≥1 slice with
  any structure in the H&N OAR set (`h&n_count.py`'s `HN_ORGANS`: parotids, spinal cord, brainstem, eyes,
  optic nerves, temporal lobe, larynx, thyroid, lens, PRV variants).
- **`hn_patients_new10.json`** → 10; **`non_hn_patients_new10.json`** → 0.
- **`parotid_train_patients.json`** → **383** patients — the train-split patients that have ≥1 parotid
  slice. Used as the **patient-level weighting list** by the local npz dataloader.
- **`dataset_split.json`** — THE authoritative split. **train 583 / val 84 / test 165 = 832 total.**

**How the split was made (`splits.py`):** load `hn_patients.json`, `random.seed(42)`, shuffle, then
70/10/20 → train/val/test. **Patient-wise** (all slices of a patient stay in one split → no data leakage).
**Locked before any training; never regenerate.** The test set was held **completely unevaluated** until
the final nnU-Net phase.

**Both-parotid presence across the split** (measured in Phase 2):
- train (583): both=186, only-R=98, only-L=99, none=200 → 383 with a parotid.
- val (84): both=22, only-R=12, only-L=13, none=37 → 47 with a parotid.
- test (165): both=44, only-R=30, only-L=28, none=63 → 102 with a parotid.

---

## 8. DATA-HANDLING UTILITIES (`dicom test/Data Handling/`)

Maintenance/analysis scripts (NOT training). All have hardcoded local paths.
- **`splits.py`** — makes `dataset_split.json` (seed 42, 70/10/20). See §7.
- **`parotid_train_patients.py`** — scans train patients, saves the 383-patient parotid list.
- **`parotid_count.py`** — counts parotid-bearing patients per split (samples first ~10 files/patient).
- **`h&n_count.py`** — classifies patients as H&N vs non-H&N by the `HN_ORGANS` set (produces the
  hn/non_hn JSONs).
- **`hn_slice_count.py`** — totals the number of slice files across all H&N patients.
- **`new_10_parotid_count.py`** — parotid presence within the new-10 batch.
- **`copy_new10_to_main.py`** — copies the new-10 DCM patients into `ML_Dataset_Final` and appends them
  to the split/HN lists (skips any already present).
- **`verify_dataset.py`** — renders per-slice verification overlays (CT + each mask as a `plt.contour`
  with an organ legend) for visual QC of a patient's npz.
- **`documentation.py`** — samples ~10,000 npz files and tallies organ-key frequency (the source of the
  "% of patients with each organ" audit numbers).

---

## 9. INTENSITY / HU HANDLING AND THE PYTORCH DATALOADERS

Two dataloader implementations exist, both a class named `ParotidDataset`, with **one important
behavioural difference** (the sample-weighting).

**Shared HU windowing (identical in both):**
```
HU     = raw_pixel * 1.0 + (-8192.0)      # to Hounsfield Units
clip   = np.clip(HU, -150, 250)           # soft-tissue window for H&N
norm   = (clip - (-150)) / (250 - (-150)) # -> [0, 1]
```
Output per item: image tensor `(1, 512, 512)` float32; mask tensor `(2, 512, 512)` float32 with
**channel 0 = PAROTID_R, channel 1 = PAROTID_L**.

**Shared augmentation:** **horizontal flip ONLY** (`random()>0.5`). Vertical flips and 90° rotations are
**deliberately excluded** — axial CT has valid left–right symmetry but up–down inversion violates the
spatial prior (spine is always at the bottom). **Critical detail:** when a horizontal flip is applied,
the **R and L mask channels are SWAPPED** (the right parotid physically moves to the left side of the
image), keeping anatomy correct.

### 9.1 `dicom test/Pytorch DataLoader Stuff/pytorch_dataloader.py` (local Mac, reads loose `.npz`)
- Globs npz per patient (handles nested `<patient>/*/data/*.npz` and flat `<patient>/data/*.npz`).
- **`get_sample_weights()` → weight 14.0** for slices whose *patient* is in `parotid_train_patients.json`,
  else 1.0. This is **patient-level** weighting. (Has a slow fallback that scans each file.)
- Used for local development / sanity checks on the Mac (needs the external drive mounted).

### 9.2 `dicom test/unet_codes_from_hpc/hdf5_dataloader.py` (HPC, reads `dataset.h5`)
- Uses `h5py.visititems` to enumerate slice groups; records **per-slice** whether it contains a parotid.
- **`get_sample_weights()` → weight 20.0 for slices that ACTUALLY contain a parotid mask**, else 1.0.
  This is **slice-level** weighting.
- **This is the version used for ALL HPC Phase-1 training.** Opens the h5 lazily per-worker.

⚠️ **The two differ:** 14.0 patient-level (local npz) vs **20.0 slice-level (HPC h5, the one actually
used)**. Both address the extreme class imbalance (parotid pixels are a tiny fraction of a 512×512 field)
via `WeightedRandomSampler(replacement=True)`.

### 9.3 `dicom test/unet_codes_from_hpc/convert_to_h5.py` (HPC only)
Packs all npz into a single `dataset.h5`. Key = `<patient>/<CT_group>/data/Z_<z>.npz`; each group stores
`image` + `PAROTID_R`/`PAROTID_L` (only — **all other organs are dropped** in the h5), **lzf** compression.
Motivation: 126,879 loose npz caused severe I/O bottleneck on HPC; the 38 GB h5 gave ~**10× faster**
loading. `dataset.h5` lives only on HPC, not in this folder.

### 9.4 DataLoader diagnostic scripts (`dicom test/Pytorch DataLoader Stuff/`)
- **`loader_check.py`** — sanity-checks one batch (shapes, value ranges, % parotid slices).
- **`hu_histogram.py`** / **`hu_verify_gem.py`** — plot/verify the HU distribution and windowing.
- **`ultimate_stat_check.py`** — comprehensive stats sweep through the loader.
Outputs live in `Pytorch Data Loader Testing/` (batch PNGs, `hu_diagnostic_fixed.png`,
`ultimate_stat_check.png`). `dicom test/train_weights_cache.npy` caches computed sample weights.

---

## 10. LOSS FUNCTIONS

### 10.1 `CombinedLoss` — `dicom test/Loss Function/loss_function.py` (Phase 1)
(Identical copy in `unet_codes_from_hpc/loss_function.py`.)
```
CombinedLoss = 0.5 * DiceLoss + 0.5 * BCELoss
```
- **DiceLoss:** applies `sigmoid` to logits, flattens **both batch and spatial** dims, computes **one
  global Dice per channel** (R, L) summed over `dim=(0,2)`, `smooth=1.0`, then averages the two channels.
- **BCELoss:** `nn.BCEWithLogitsLoss` (raw logits).
- **Sigmoid, not softmax** — left and right parotids are treated as **independent binary problems**, not
  mutually exclusive classes. Returns `(total, dice, bce)` for logging.

### 10.2 `MaskedPartialLoss` — `pipeline/masked_loss.py` (Phase 2, for future multi-organ)
Independent sigmoid channels + **masked** (BCE + Dice), for partial-label multi-organ training. A per-sample
`mask [B,C]` marks which classes are actually annotated; **only annotated (sample,class) pairs contribute**
to the loss, so the network is never penalised for correctly predicting an organ the clinician simply chose
not to contour (the annotation-gap problem, §16). Works for 2D `[B,C,H,W]` or 3D `[B,C,D,H,W]` (spatial dims
reduced dynamically). **A bug was found and fixed during review:** an earlier draft let masked-out classes
leak into the averaged Dice term; the fix applies the mask inside the Dice reduction
(`dice_loss = 1 - (dice*mask).sum()/valid_classes`). Has a `__main__` **unit test** proving that corrupting
an unannotated channel leaves the loss identical, and that an all-annotated mask equals the plain combined
loss. **Not yet used in training** — reserved for the multi-organ phase.

---

## 11. PHASE 1 — THE FOUR FROM-SCRATCH ARCHITECTURES

All four are **hand-implemented in PyTorch from scratch** (no MONAI/SimpleITK/segmentation frameworks),
each **verified against its source paper** before training. All take `in_channels=1, out_channels=2` and
output **raw logits** (sigmoid applied in the loss). Trained **2D slice-level** (HPC VRAM limit). Code
lives per-model in `dicom test/<model>_codes_from_hpc/` (the HPC versions actually run) and local variants
in `dicom test/Unet-Mac/` and `dicom test/Trans UNet Mac/`; mirrored under
`…/03_models/from_scratch/{unet,attention_unet,transunet,swin_unet}/`.

### 11.1 U-Net (baseline) — `unet.py`
- **Paper:** Ronneberger et al. 2015. **~31M params.**
- `DoubleConv` = (Conv3×3 → BatchNorm → ReLU) ×2 (BN added vs the original paper; `padding=1` "same"
  to preserve 512×512). `Encoder` = DoubleConv + MaxPool2d (returns pooled + pre-pool skip).
  `Decoder` = ConvTranspose2d 2× → concat skip → DoubleConv. Features `[64,128,256,512]` + `1024`
  bottleneck. Final 1×1 conv → 2 channels, no activation.

### 11.2 Attention U-Net — `attention_unet.py`
- **Paper:** Oktay et al. 2018. **~31.4M params.**
- Identical U-Net backbone (`ConvBlock`, channels 64→1024) **plus `AttentionBlock` gates on all 4 skip
  connections**. Each gate: gating signal `g` from the decoder (coarser) + skip `x` from the encoder →
  1×1 conv projections to `F_int = F_g/2` → element-wise add → ReLU → 1×1 conv → **sigmoid attention map**
  → multiply with the skip. Suppresses irrelevant skip features.

### 11.3 TransUNet — `transunet.py`
- **Paper:** Chen et al. 2021 (arXiv:2102.04306). **~102.5M params.** Hybrid CNN–Transformer.
- **`ResNet50Encoder`** (first 3 stages): stem (Conv7×7 s2 → BN → ReLU → MaxPool) → 128×128;
  **stage1 = 3 bottleneck blocks** (→256ch, 128×128); **stage2 = 4 blocks** (→512ch, 64×64);
  **stage3 = 6 blocks** (→1024ch, 32×32). Skips taken from stage1 (256ch/128×128) and stage2 (512ch/64×64).
  ⚠️ **Code is authoritative: stage3 = 6 blocks** (some prose said 9).
- **`PatchEmbedding`**: 1×1 conv projecting the 1024ch/32×32 map to hidden 768 → 32×32=**1024 tokens** +
  learned positional embeddings.
- **`TransformerBlock`**: 12 layers, 12 heads, MLP 3072, GELU, **pre-norm** (`norm_first=True`, matches ViT).
- **CUP decoder** (`DecoderCUPBlock`): bilinear 2× upsample → concat skip → 3×3 conv-BN-ReLU ×2. Levels:
  512→(skip2)→256@64×64 → (skip1)→128@128×128 → 64@256×256 → 32@512×512. 1×1 seg head → 2ch.
- Kaiming/trunc-normal init. **Trained from scratch, no ImageNet pretraining** (controlled comparison),
  input adapted to 512×512 single-channel.

### 11.4 Swin-UNet — `swin_unet.py`
- **Paper:** Cao et al. 2021 (arXiv:2105.05537, ECCVW 2022). **~27M params.** **Pure transformer**, zero
  conv in the backbone.
- `PatchEmbedding`: 4×4 patches → 96-dim (512×512 → 128×128 tokens). `PatchMerging` (2×2 concat → linear)
  downsamples; `PatchExpanding` upsamples. `WindowAttention` = W-MSA + shifted-window (SW-MSA) with
  **relative position bias**; Swin blocks 2/stage; channels 96/192/384/768.
- ⚠️ **Key adaptation:** the paper's final `FinalPatchExpanding` (128×128→512×512 in one step) creates a
  (B, 262144, 96) tensor that **OOM'd** the shared GPU. **Replaced with a linear seg head at 128×128 +
  bilinear interpolation to 512×512** — a common practical workaround. Trained from scratch (no ImageNet
  Swin-T weights).

### 11.5 nnU-Net v2 — NOT a from-scratch architecture
Handled by the official `nnunetv2` framework in Phase 2 (Section 14), documented separately. In Phase 1's
era it was listed as "pending" (never trained then, due to HPC GPU contention).

---

## 12. PHASE 1 — TRAINING CONFIG, SCRIPTS, PBS JOBS, EVALUATION FRAMEWORK

### 12.1 Identical training config across all 4 models (for a fair comparison)
| Parameter | Value |
|-----------|-------|
| Optimiser | Adam, lr **1e-4** |
| LR scheduler | `ReduceLROnPlateau(mode='min', patience=5, factor=0.5)` on val_loss |
| Early stopping | patience **10** epochs |
| Batch size | **8** |
| Max epochs | **50** |
| Mixed precision (AMP) | autocast + GradScaler for Attention/Trans/Swin (U-Net baseline ran without) |
| Sampler | `WeightedRandomSampler(replacement=True)`, num_workers=4 |
| Checkpoint | best (lowest val_loss) saved as `{epoch, model_state_dict, optimizer_state_dict, val_loss}` |
| History | per-epoch `{epoch,train_loss,train_dice,val_loss,val_dice}` JSON, written every epoch |

**Training scripts** (`<model>_train.py`): standard loop; `train_one_epoch` + `validate`. **The validation
loop skips batches where `masks.sum()==0`** so empty slices don't dominate val_loss. The Attention/Trans/
Swin scripts add AMP and **checkpoint-resume** (reload model/opt/scheduler/scaler, continue from saved
epoch, keep history) so a 48-h walltime kill doesn't lose progress.

### 12.2 PBS job scripts (`.sh`)
- Training jobs: `#PBS -l select=1:ncpus=4:ngpus=1:mem=40gb`, `walltime=48:00:00`, queue `gpu`,
  `source /apps/anaconda3/bin/activate deeplearning`, `python3 -u <train>.py`. Use
  `exec > logfile 2>&1` + `-u` for **live logging** (PBS doesn't flush `-o/-e` until job end).
  TransUNet job exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (OOM fragmentation safeguard).
- Eval jobs: shorter walltime, queue `workq`, **CPU** (`ncpus=8`, no GPU) — forced by GPU contention.

### 12.3 Evaluation framework (`<model>_evaluate.py`, e.g. `attention_evaluate.py`)
- **3D patient-level volumetric** metrics on the **VALIDATION set** (84 patients; only the ~43 with parotid
  annotations counted). Slices stacked into 3D volumes per patient; predictions `sigmoid` then threshold
  **0.5**; metrics per-structure (R, L) then averaged.
- **Metrics:** 3D Dice; **Clinical Tversky (α=0.3, β=0.7)** — asymmetric, penalises false-negatives (under-
  segmentation) more, reflecting the clinical priority of not missing tissue; **HD95 (mm)** — 95th-pctile
  Hausdorff via `distance_transform_edt` + `binary_erosion`, `PIXEL_SPACING_MM=0.977`, `nanmean` aggregation;
  **Surface Dice @3mm tolerance**.
- **Edge cases:** both pred & GT empty → Dice/SurfDice = 1.0, HD95 = 0; only one empty → HD95 = NaN
  (excluded via `nanmean`). Patients with no parotid GT are skipped. Results appended to `results_tracker.csv`.
- ⚠️ **This Phase-1 eval uses isotropic 0.977 pixel spacing for HD95** (2D-stacked). The Phase-2
  `eval_testset.py` uses *true anisotropic* spacing (0.977×0.977×3.0) — so Phase-1 and Phase-2 boundary
  numbers are not strictly comparable.
- **Prediction visualisers:** `check_prediction.py` (U-Net confidence heatmap on a random val slice),
  `<model>_predict.py` (Swin/Trans) render qualitative overlays into the `Predictions/` folders.

---

## 13. PHASE 1 — RESULTS

### 13.1 Two different "Dice" numbers — do not conflate
- **Headline 3D volumetric val Dice (~0.62)** — from the evaluation framework (§12.3), in
  `documentation/results_tracker.csv` / `04_results/results_tracker_VALSET_fromscratch.csv`.
- **Per-epoch `val_dice` (~0.33)** — a *different, 2D slice-level* quantity in the training-history JSONs.
  Lower by construction. **When someone cites "0.62" they mean the volumetric metric; "0.33" is the
  training-log curve.** Don't mix them.

### 13.2 Validation-set 3D volumetric results (test set was still locked at this stage)
| Model | 3D Dice | Tversky | HD95 (mm) | Surface Dice (3mm) | Params | Best epoch |
|-------|---------|---------|-----------|--------------------|--------|-----------|
| Vanilla U-Net | 0.6209 | 0.6247 | 4.74 | 0.7217 | ~31M | 4/14 |
| Attention U-Net | **0.6346** | 0.6440 | 6.83 | 0.7296 | ~31.4M | 10/19 |
| TransUNet | 0.6332 | 0.6423 | **3.58** | **0.7379** | ~102.5M | 14/23 |
| Swin-UNet | 0.5106 | 0.5170 | 12.10 | 0.6269 | ~27M | 13/22 |

### 13.3 Per-epoch training-history highlights (from the JSON/CSV logs)
- **U-Net:** 14 epochs, best val_loss **0.1685 @ epoch 4** (val_dice≈0.334), early-stopped @14.
- **Attention U-Net:** history recorded **epochs 3–19** (interrupted after epoch 2 to add AMP+resume; the
  `attention_history.csv` starts at epoch 3), best val_loss **0.1637 @ epoch 10**, early-stopped @19.
- **TransUNet:** 23 epochs, best val_loss **0.1692 @ epoch 14**, higher volatility (transformer, no pretrain).
- **Swin-UNet:** 22 epochs, best val_loss **0.2101 @ epoch 13** — clearly worse than the CNN/hybrid models.

### 13.4 The Phase-1 findings (documented and important)
1. **CNN/hybrid models cluster at 0.62–0.63 Dice regardless of param count (31M → 102M).** This diagnoses
   a **performance ceiling from the data/preprocessing (raw 2D slices, mixed spacing, no resampling, single
   HU window)**, not architectural capacity. All converge to ~0.16–0.17 val loss.
2. **Pure transformer underperforms without pretraining:** Swin-UNet (0.5106) trails all CNN-containing
   models — 513 parotid patients is too little for a from-scratch ViT.
3. **TransUNet has the best boundaries** (HD95 3.58, SurfDice 0.7379): the ResNet-50 local features + 12-layer
   ViT global context preserve boundary precision best.
4. **Attention U-Net: better overlap, worse boundary** than U-Net (HD95 6.83 vs 4.74) — attention gates
   occasionally produce small isolated prediction islands several mm off, barely hurting Dice but hurting HD95.

**This ceiling finding is what motivated Phase 2** (move to nnU-Net's full 3D self-configuring pipeline).

---

## 14. PHASE 2 — nnU-Net v2: THE FULL 3D PIPELINE

**Decision (July 2026):** backbone = **nnU-Net v2** (self-configuring SOTA medical-segmentation framework),
**from scratch** (cross-dataset pretrained-weight transfer judged too finicky; nnU-Net from scratch is
already SOTA). The 4 from-scratch models are kept as portfolio evidence, not the deliverable. Two friend
suggestions were **evaluated and rejected**: (a) **Mixture-of-Experts** for dataset variation — rejected,
heterogeneity is better handled by resampling+normalisation+augmentation (which nnU-Net does; its built-in
5-fold ensemble captures the useful "multiple experts" benefit); (b) **quadrant-tiling each image into 4
sub-images** — rejected, fixed quadrants bisect organs and lose context; sliding-window patching (nnU-Net)
is the principled equivalent.

**Staged plan:** Phase 1 = nnU-Net **parotid** model + test eval + demo; Phase 2 (future) = **multi-organ**
with the masked partial-label loss.

All Phase-2 pipeline code is in **`pipeline/`** (mirror in `…/03_models/nnunet/code/`).

### 14.1 `pipeline/build_volumes.py` — per-slice npz → 3D NIfTI
- For a patient, groups slice files by CT group; **picks the CT group with the most target-organ
  annotations** (tie-break: slice count). Sorts slices by **physical Z parsed from the filename**.
- **Z spacing = median of the filename-Z differences** (typically 3.0 mm); warns on non-uniform spacing,
  gaps (missing slices), or duplicate Z. **In-plane spacing = 0.977 mm** (documented assumption — not
  stored in npz). Stacks image (int16) + per-organ binary volumes into `(Y, X, Z)`.
- `save_nifti` writes `<prefix>_0000.nii.gz` (image) + `<prefix>_<organ>.nii.gz`, affine = diag spacing.
- **DATA FACT established here:** npz store no spacing; Z from filenames; in-plane assumed 0.977.

### 14.2 `pipeline/make_nnunet_dataset.py` — build the nnU-Net raw dataset
- Produces `Dataset<ID>_Parotid/` with `imagesTr/labelsTr` (train+val patients) and `imagesTs/labelsTs`
  (the **locked TEST patients**, kept for the one-time final eval — nnU-Net never trains on them).
- **Integer labelmap: 0=bg, 1=parotid_r, 2=parotid_l** (L drawn last). **Only patients with BOTH parotids
  annotated are included** (keeps the labelmap clean; single-side patients deferred to the future masked-loss
  phase). Writes `dataset.json` and **`case_mapping.json`** (PARxxxx → real patient id + annotation flags).
- **Dataset001_Parotid** (original build): **208 train+val** (186+22 both-annotated) + **44 test**.

### 14.3 `pipeline/fix_labels_qc.py` — the corrected dataset (Dataset002)
Builds a corrected dataset from an existing one:
1. **L/R consistency:** reassign the two parotid labels **by geometry** — smaller **axis-1 (column)
   centroid = R = label 1**, larger = L = label 2. Only the label *id* is swapped where needed; contour
   shapes are untouched.
2. **QC:** drop any case whose gland volume **> 3× the dataset median** (threshold 18,681 voxels) — catches
   corrupt/blown-up masks (e.g. **PAR0240 = patient `1~240067`**, 34,963 voxels).
- Images cloned via hardlink (instant, no extra disk).
- **Dataset002_Parotid build result:** train **kept 208, dropped 0, L/R-corrected 1**; test **kept 43,
  dropped 1** (PAR0240), L/R-corrected 0. **252 total cases: 251 kept, 1 dropped, only 1 lr_flipped.**
  (This "only 1 flip" is the crucial second finding — see §15.)

### 14.4 `pipeline/nnUNetTrainer_250epochs_noMirror.py` — the custom trainer
Subclasses `nnUNetTrainerNoMirroring`, sets `num_epochs = 250`, **mirror augmentation DISABLED**. Must be
copied into the nnU-Net package's `training/nnUNetTrainer/variants/` directory to be discoverable. **The
`__init__` must use the explicit `(plans, configuration, fold, dataset_json, device)` signature — NOT
`*args/**kwargs`** (nnU-Net records init kwargs by iterating the signature; `*args` yields names not in the
frame → `KeyError: 'args'`). Predict with `--disable_tta`.

### 14.5 `pipeline/eval_testset.py` — the held-out test evaluator
Same 4 metrics as Phase 1 (Dice / Clinical Tversky α0.3 β0.7 / HD95 / Surface-Dice @3mm), but with **TRUE
anisotropic spacing `(0.977, 0.977, 3.0)`** so boundary metrics are physically correct, and **bbox-cropping**
for speed. Compares predicted labelmaps (1=R,2=L) vs `labelsTs`, per-structure then averaged; skips a side
with no GT (doesn't inflate); `nanmean` for HD95. Validated: perfect pred → Dice 1.0/HD95 0; a 4-voxel shift
→ Dice 0.68/HD95 3.82 mm (≈4×0.977, physically correct). Appends to a results CSV.

### 14.6 nnU-Net auto-configuration (what it chose for 3d_fullres)
From `plans.json`: **spacing [3.0, 0.977, 0.977]** (anisotropy preserved, not forced isotropic),
**patch size [48, 224, 192]**, **batch size 2**, **`CTNormalization`**, `InstanceNorm3d`, PlainConvUNet,
**6 stages, features [32,64,128,256,320,320]** (≈30M params — same order as the owner's U-Net, far smaller
than the 102M TransUNet). Fold 0 internal split: 166 train / 42 val.

### 14.7 `pipeline/test_and_visualize.py` — inference + human-viewable rendering
Runs the trained 3D nnU-Net (mirroring off) on a folder of scans (or reuses saved predictions), draws 2D
axial montages (pred R=red / L=blue fills, GT dashed) + optional 3D marching-cubes gland surface renders +
per-case metrics + an `index.html` gallery. Input modes: `--images` (nnU-Net NIfTI) or `--patients`
(reconstruct npz on the fly via `build_volumes`). This produced the `parotid_viz_testset` / `parotid_viz_ensemble`
galleries.

### 14.8 Compute logistics
HPC OOM'd (both L40s saturated) → rented **RunPod H100** for the single-fold run (~15 s/epoch, ~1 h),
then a **4× 4090/A40** box for folds 1–4 in parallel. Dataset relayed HPC→Mac→pod (firewall). Custom trainer
must be re-registered on every fresh pod. Every fold trained with `--npz` (needed for ensembling/postprocessing).
Runbooks: `pipeline/README_HPC.md`, `README_CLOUD_retrain.md`, `README_CLOUD_5fold.md`, and
`Parotid-Project/Docs/HANDOFF_5fold_ensemble.md` / `RUN_patient_on_gpu.md`.

---

## 15. THE L/R MIRROR BUG — THE FULL DIAGNOSTIC STORY (the project's best result)

This is the strongest engineering/debugging story in the project. Sequence of events:

1. **First nnU-Net (Dataset001, default trainer) test result was WORSE than the baselines:**
   **3D Dice 0.5030, HD95 65.03 mm** (nnU-Net's own internal fold-0 val Dice was 0.4538). A red flag, since
   training looked perfectly healthy (train/val loss tracked together, pseudo-Dice ~0.70–0.74).
2. **Per-case diagnostic:** the per-side Dice distribution was strongly **bimodal** — a cluster at 0.80–0.91
   and a separate cluster near 0. Median (0.595) > mean (0.503): a subset of catastrophic cases dragging it
   down, not a uniform weakness. 28/88 sides < 0.3.
3. **L/R swap test:** for each case, compute normal Dice vs **swapped** Dice (pred-R vs GT-L, pred-L vs GT-R).
   **11 of 44 cases scored 0.6–0.85 when swapped vs ~0 normally.** Taking better-of-normal/swap lifts the
   mean 0.503 → 0.661. Swap-affected cases occurred in **both** pipelines (9 WC, 2 DCM) — not a parse issue.
4. **Merged (L/R-agnostic) Dice** (score parotid tissue ignoring which side): **mean 0.820, median 0.841,
   all 44 cases ≥ 0.60, 0 cases < 0.30.** → The model **finds both glands well**; only the L/R *naming* is
   wrong on a subset.
5. **Ruled out the labels being flipped:** the `fix_labels_qc.py` geometric relabel corrected **only 1 of
   251 cases**. If GT had been flipped on ~25% of patients, it would have corrected ~60. It corrected 1. →
   **The ground-truth annotations are geometrically consistent (R ≈ smaller axis-1 centroid everywhere);
   the flips are produced by the MODEL, not present in the data.**
6. **Root cause:** **nnU-Net's default left-right mirror data augmentation + test-time mirroring
   (`inference_allowed_mirroring_axes (0,1,2)`)** makes left/right-distinct structures interchangeable — a
   known-but-easy-to-miss failure mode. **Isolated it:** `--disable_tta` *alone* only recovered 0.503 → 0.564
   (HD95 still ~64 mm), proving the confusion was **baked into the weights**, not just TTA.
7. **Fix:** retrain with the custom **`nnUNetTrainer_250epochs_noMirror`** (mirroring disabled) + predict
   with `--disable_tta`.

### 15.1 Before vs after (held-out test set)
| Metric | Mirror-ON (original) | Mirror-OFF (fixed) |
|---|---|---|
| nnU-Net internal fold-0 val Dice | 0.4538 | **0.7967** |
| Test per-side 3D Dice | 0.5030 | **0.8187** |
| HD95 | 65.03 mm | **5.25 mm** |
| Surface Dice (3mm) | 0.5816 | **0.8965** |

### 15.2 Final results table (held-out TEST set, 43 both-parotid cases; corrupt PAR0240 dropped)
| Scoring / model | 3D Dice | Tversky | HD95 (mm) | Surface Dice (3mm) | Source |
|---|---|---|---|---|---|
| Mirror-ON single fold | 0.5030 | 0.5028 | 65.03 | 0.5816 | `results_tracker_TEST.csv` |
| **noMirror single fold** | **0.8187** | 0.8166 | **5.25** | 0.8965 | `results_tracker_TEST.csv` (R 0.8278 / L 0.8095) |
| **noMirror 5-fold ENSEMBLE (final)** | **0.8202** | 0.8174 | **5.24** | **0.8990** | `results_5fold_TEST.csv` (R 0.833 / L 0.808) |

**Headline improvement of the whole project: 0.62 (Phase-1 2D val) → 0.82 (Phase-2 3D test), HD95 65 mm →
5.2 mm.** The 0.82 is comparable to published parotid-segmentation results (~0.85). (Caveat: Phase-1 0.62 is
a *validation* number, Phase-2 0.82 is a *test* number — not strictly apples-to-apples, but the 3D pipeline
is clearly the driver.) Per-case metrics for all 43 test cases live in
`Parotid-Project/Visualizations/parotid_viz_testset/metrics.csv` (single fold) and
`…/parotid_viz_ensemble/metrics.csv` (ensemble), and ranked in `Website/_parotid-render-tools/ranked.csv`
(best case PAR0228 ≈ 0.916; worst PAR0223 ≈ 0.577, dragged by an L-side HD95 of 21 mm). One real un-seen
patient (260594) was also run end-to-end: Dice 0.866.

### 15.3 A quick alternative that was considered but not shipped
Before choosing to retrain, the owner was offered "ship-as-is with post-processing" — split the merged
prediction into two **connected components** and assign L/R by axis-1 centroid — which recovered per-side
**R 0.829 / L 0.801 / overall 0.815** on the existing (mirror-on) model at zero extra compute. The owner
chose the **clean retrain (noMirror)** instead, which gives the model correct native L/R output. Connected-
component largest-island post-processing remains available as a cheap HD95 cleanup.

---

## 16. CLINICAL FINDING (ANNOTATION GAP) + MULTI-ORGAN AUDIT/PLAN

### 16.1 The annotation-gap clinical finding (on record)
During review, models consistently predicted **bilateral** parotids where the ground truth had only
**unilateral** annotation. Initially suspected as model hallucination — but **The clinical advisor confirmed the
predictions were anatomically correct.** When a tumour is strictly one-sided, clinicians **deliberately do
not contour the contralateral (healthy-side) parotid** because it isn't at risk in that treatment plan
(saves ~15 min/structure). This is a **deliberate clinical decision, not an annotation error.** Consequences:
- **Fully-absent contours** don't inflate metrics: Dice = 1.0 (correct negative), HD95 = NaN (excluded).
- **Partially-annotated cases** (clinician started but stopped early) **penalise stronger models** — a model
  that correctly extends beyond the incomplete GT gets distance penalties for correct anatomy.
- → **Reported Dice is a conservative underestimate for the stronger models.** And any multi-organ loss
  **must mask out un-annotated organs** (hence `masked_loss.py`).

### 16.2 The July-2026 data audit (`documentation/DATA_AUDIT_AND_PLAN_2026-07.md`)
A 150-patient random sample of `ML_Dataset_Final`. Central constraint: **annotation is partial and
inconsistent — a missing mask does NOT mean the organ is absent.** Reliably-annotated H&N OARs (share of
patients with ≥1 positive slice):

| Structure | % patients | Note |
|---|---|---|
| EYE_L / EYE_R | 65% / 64% | good bilateral target |
| OPTIC_NERVE_L / _R | 65% / 15% | R under-annotated |
| TEMPORAL_LOBE | 61% | easy, optional |
| SPINAL_CORD_PRV / SPINAL_CORD | 54% / 41% | key OAR |
| LARYNX | 51% | good target |
| PAROTID_L / PAROTID_R | 46% / 45% | **primary clinical target** |
| OPTIC_CHIASMA | 42% | tiny, hard |
| LENS_R / LENS_L | 59% / 15% | very small, hardest |
| BODY | 35% | trivial outline |
| PTV (tumour) | 92% | target volume, not an OAR — handle separately |
| BRAINSTEM | 5% | too rare to model reliably |

Thorax/abdomen structures (LUNG, LIVER, BREAST, FEMUR, KIDNEY, HEART) appear only because non-H&N patients
are mixed in — exclude from the H&N model. **Verdict: multi-organ is feasible.** Proposed ~8-organ target
set: parotids L/R, eyes L/R, spinal cord, larynx, optic nerves L/R (+ optic chiasm / temporal lobe optional).
**Solutions the audit endorses:** resample to common spacing + intensity-normalise + augment (nnU-Net does
this — solves heterogeneity); masked/marginal loss for partial labels; keep the 12-patient exclusion + add
automated area/eccentricity QC; run the locked 165-patient test **once** at the very end.

---

## 17. CHECKPOINTS, CAVEATS, DISCREPANCIES, AND WHAT-RUNS-WHERE

### 17.1 Checkpoints present locally (`checkpoints/`)
| Model | File | Size | Date |
|---|---|---|---|
| U-Net | `unet/best_model_unet.pth` | 372 MB | Mar 18 |
| Attention U-Net | `attention_unet/best_model_attention.pth` | 377 MB | Mar 24 |
| Attention U-Net | `attention_unet/best_model_final_attention.pth` | 377 MB | Mar 19 |
| TransUNet | `transunet/best_model_transunet.pth` | 1.23 GB | Mar 24 |
| Swin-UNet | `swin_unet/best_model_swinunet.pth` | 336 MB | Mar 24 |

Each model dir also has `training_history_*.json`; the Attention dir has `attention_history.csv` +
`attention_learning_curves.png`. **Two Attention checkpoints exist** (`best_model_attention.pth` Mar 24 and
`best_model_final_attention.pth` Mar 19).

**nnU-Net checkpoints are NOT in this local folder in full** — they were downloaded to the owner's
`~/Desktop/` (e.g. `nnunet_results_Dataset002_noMirror/…/fold_0/checkpoint_final.pth` (234 MB) +
`checkpoint_best.pth`, and the 5-fold `nnunet_results_Dataset002_5fold/`). What IS committed under
`Parotid-Project/…/03_models/nnunet/` and `Parotid-Project/Results/` are the **config/log artifacts**
(`dataset.json`, `plans.json`, `dataset_fingerprint.json`, `fold_*/debug.json`, training logs) for
`trained_model_v1_mirror_original` and `trained_model_v2_noMirror_FINAL`, plus the `Dataset001/002`
`case_mapping*.json`. The pod was only *stopped* (not deleted) when balance ran out; `/workspace` survived
and the final checkpoint was recovered.

> **⚠️ CORRECTION (2026-07-17) — §17.1 and §18 below are STALE on where the data/checkpoints live.** As of
> the ablation study, the following ARE present locally (verified on the Mac), contradicting the "not in this
> folder / JSON-only" claims above and in §18:
> - **`Parotid-Project/Datasets/`** holds the **full Dataset001 (6.6 GB) and Dataset002 (6.5 GB)** with real
>   `imagesTr/labelsTr/imagesTs/labelsTs` — not JSON-only.
> - **`Parotid-Project/Results/`** holds the **nnU-Net checkpoints including the 5-fold ensemble**.
> - **Ablation checkpoints/datasets/predictions** are under `ablation_study/_pod_results/` (830 MB, Track A:
>   2d + dirty-430 + gapped-208 nnU-Net models), `ablation_study/_pod_results_B/` (1.6 GB, Track B: E4 custom
>   3D U-Net, E3 pretrained TransUNet + Swin, P0 preds, A1 figures), `ablation_study/E2_annotation_gap/nnUNet_raw/`
>   (Dataset003_ParotidDirty=430, Dataset004_ParotidGapped=208), and `ablation_study/E3_transformer_pretrain/data/`
>   (10.6 GB `trainval.h5`). All checkpoints were verified to load. **Both training pods are terminated; no
>   compute is pending.**
> - New root docs since the study: **`POD_UPLOAD_PLAYBOOK.md`** (Mac→RunPod transfer lessons) and the
>   `ablation_study/` folder. See §22 for the full artifact map.

### 17.2 Known discrepancies / gotchas (keep these in mind)
1. **Patient counts vary between docs:** V1 doc = 834 HN / 480 parotid / 904 total; V2+brain = 844 HN /
   513 parotid. **Machine files are authoritative:** 844 HN, 383 in parotid-train list, splits total 832.
2. **Sample-weight value differs:** local npz loader = **14.0 patient-level**; HPC h5 loader = **20.0
   slice-level**. **HPC training used 20.0.**
3. **`unet_codes_from_hpc/unet_train.py`** historically hardcoded `os.environ["CUDA_VISIBLE_DEVICES"]="1"`
   (the OOM-causing line the docs say was later removed; eval `.sh` scripts instead auto-pick the emptiest
   GPU via `nvidia-smi … sort -nr`).
4. **TransUNet ResNet stage-3 = 6 blocks in code** (some prose said 9). Code is source of truth.
5. **Stale hardcoded paths** everywhere — `/Users/ritvikmod/Desktop/ct scan project/`,
   `/Users/ritvikmod/Desktop/Projects/ct scan project/`, and HPC `/home/<hpc-user>/project/`. The folder
   is now at `…/Desktop/Academics/Projects/ct scan project/`. Fix before running.
6. **`dataset.h5`, NIfTI conversions, nnU-Net preprocessing/checkpoints are HPC/pod/external-drive only** —
   not reproducible locally unless the npz data is mounted and scripts re-run.
7. **Headline Dice caveat:** ~0.62 = Phase-1 3D *volumetric validation*; ~0.33 = Phase-1 2D per-epoch
   `val_dice`; 0.82 = Phase-2 3D *test*. Three different quantities. **⚠️ SUPERSEDED by §20 (2026-07-17):**
   the 0.62-vs-0.82 comparison itself is invalid because it compares a **val** number to a **test** number —
   on the SAME locked test set Phase-1 scores **0.7434**, so the honest improvement is **0.7434 → 0.8187
   (+0.075)**, and ~62% of the famous "+0.20" was a val-vs-test artifact. **Never quote 0.62 → 0.82 again.**
8. **`unet.py` U-Net baseline ran without AMP** (only Attention/Trans/Swin used AMP).
9. **`documentation/ct_scan_project_file_map.md` mislabels `train_weights_cache.npy` as "per-class loss
   weights" — it is not.** It is a per-**slice** `WeightedRandomSampler` weight vector (used in *sampling*, not
   in the loss): verified shape **(78231,)**, values **{1.0: 71297, 14.0: 6934}** (14× up-weighting on the
   parotid-bearing slices — the npz loader's value; the h5/HPC loader uses **20×**, see gotcha 2). It biases
   *which slices are drawn*, and has nothing to do with class weighting inside the loss.

### 17.3 What can run where
- **Local Mac (no HPC):** read/refactor/debug any code; fix paths; write new architectures/loaders; run
  **inference + visualisation** on the on-disk `.pth` checkpoints (CPU/MPS; slow for TransUNet but fine for
  spot checks) — **requires the external drive mounted** for real slices. Re-run `convert_to_h5.py`,
  `splits.py`, counting/verification utilities (all need the npz mounted).
- **Requires HPC / cloud GPU:** full training of any of the 4 models or nnU-Net (needs L40/H100/4090-class
  GPUs); anything touching `dataset.h5`, NIfTI conversion, or nnU-Net planning/preprocessing artifacts.
- Claude/agents on the Mac cannot reach HPC directly; HPC jobs go via PBS on `rachel`, results pulled back;
  cloud jobs go via SSH to a RunPod pod.

---

## 18. COMPLETE FILE-SYSTEM MAP

Root: `ct scan project/`

**Datasets (bulk; present only when the external drive/HPC copy is available):**
- `ML_Dataset_Final/` — 914 patient dirs. **Canonical training dataset.** `<patient>/[<CT_group>/]data/Z_*.npz`
  + `verification/*.png`.
- `ML_Dataset_Master/` — 797 entries (WC-parsed master).
- `ML_Dataset_Master_DIRECT_DCM/` — 107 entries (DICOM-RT parsed).
- `ML_Dataset_Master_DIRECT_DCM_new/` — 10 entries (new-10 batch, IDs 260251–260275).
- `raw patient data/` — raw Elekta source folders (`dummy patient data/` + `new 10 patient data/`).

**Code & artifacts:**
- `dicom test/` — **MAIN Phase-1 code directory.** Subfolders: `Data Handling/` (utilities §8),
  `Direct DCM/` (DCM parsers §5.2), `Loss Function/` (`loss_function.py` §10.1), `Parser/` (WC parsers §5.1),
  `Pytorch DataLoader Stuff/` (npz loader + diagnostics §9), `Unet-Mac/` + `Trans UNet Mac/` (local model
  variants), `unet_codes_from_hpc/` / `attention_unet_codes_from_hpc/` / `swinunet_codes_from_hpc/` /
  `transunet_codes_from_hpc/` (HPC snapshots: `<model>.py`, `<model>_train.py`, `<model>_evaluate.py`/
  `_predict.py`, `.sh` PBS jobs; plus `hdf5_dataloader.py`, `convert_to_h5.py`, `check_prediction.py` in the
  unet folder). `train_weights_cache.npy` = cached sample weights.
- `pipeline/` — **Phase-2 nnU-Net code** (§14): `build_volumes.py`, `make_nnunet_dataset.py`,
  `fix_labels_qc.py`, `masked_loss.py`, `eval_testset.py`, `nnUNetTrainer_250epochs_noMirror.py`,
  `test_and_visualize.py`, and READMEs (`README_HPC.md`, `README_CLOUD_retrain.md`, `README_CLOUD_5fold.md`).
- `checkpoints/` — the 5 Phase-1 `.pth` + histories (§17.1).
- `Predictions/` — Phase-1 qualitative prediction PNGs: `Basline UNet Predictions/` (10),
  `TransUNet Predictions/` (5), `SwinUNet Predictions/` (2), `Attention UNet Predictions/` (1).
- `patient classification/` — split JSONs (§7). `label standardisation/` — standardisation logs (§6).
- `Pytorch Data Loader Testing/` — loader-dev outputs (batch PNGs ×4 sets, HU/stat diagnostics).
- `day 1 parser building /` — earliest exploration (`dicom_overlay_test.png`, `organ_shapes.png`).
- `sanity_check_outputs/` — `sanity_check_inference.py` outputs: per-slice overlays, `demo_preview_*.png`,
  `volume_recon_check.png`, and **`lr_swap_diagnosis.png`** (the L/R bug figure).
- `sanity_check_inference.py` (root) — loads the Attention checkpoint, confirms it loads and produces
  anatomically-correct predictions (per-slice 2D Dice 0.53–0.90); confirms preprocessing recipe.
- `documentation/` — the written docs: **`AI_BRAIN_COMPLETE_CONTEXT.md`** (April-2026 narrative brain),
  **`CASE_FILE_nnUNet_Parotid_2026-07-07.md`** (the nnU-Net + L/R diagnosis case file),
  **`DATA_AUDIT_AND_PLAN_2026-07.md`**, `nnunet_noMirror_TEST_result.md`, `ct_scan_project_file_map.md`,
  `results_tracker.csv`, plus `HN_Segmentation_Project_Documentation_V1/V2` (docx/pdf/pages) and
  `Dataset_Overview.pdf`.
- **`Parotid-Project/`** — a **later, self-contained restructure** of the whole project (July 2026):
  - `Docs/` — `PROJECT_STATE_FOR_AI_AGENT.md` (factual per-file dump), `RESUME_CONTEXT_Parotid_Project.md`
    (CV brief), `HANDOFF_5fold_ensemble.md`, `RUN_patient_on_gpu.md`.
  - `Datasets/Dataset001_Parotid/` + `Dataset002_Parotid/` — nnU-Net `dataset.json` + `case_mapping*.json`.
  - `Results/` — nnU-Net config/log artifacts for the mirror/noMirror/5fold runs, `results_*_TEST.csv`,
    `pred_*` config dumps, `patient_260594_result/`.
  - `Segmentation_Complete/Parotid_Segmentation_Complete/` — a **clean self-contained mirror** of the entire
    pipeline organised as `01_parsers/ 02_preprocessing/ 03_models/ 04_results/ 05_inference_demo/ docs/
    sample_data/` (2 held-out NIfTI volumes to run immediately). `_build_manifest.sh` records what was copied.
  - `Visualizations/` and `Website/` — **showcase artifacts (see scope note; ignore).**

**Showcase / presentation artifacts (intentionally out of core scope — listed for completeness only):**
`webapp/` (FastAPI), `demo/` (Gradio + `examples/*.npz`), `Parotid-Project/Website/` (`parotid-site`, `gpt`,
`_parotid-readingroom-backup`, `_parotid-render-tools`), `Parotid-Project/Visualizations/`
(`Parotid_AI_Showcase.html`, viz galleries), and the `05_inference_demo/webapp` + `gradio_demo` mirrors.
These are throwaway presentation experiments the owner asked to drop from working context.

---

## 19. GLOSSARY OF PROJECT-SPECIFIC TERMS

- **OAR** — Organ At Risk (healthy tissue to spare during radiotherapy).
- **Parotid gland** — large salivary gland beside the ear; the primary target of this project.
- **Xerostomia** — chronic dry mouth from parotid over-dose; the harm this project helps avoid.
- **IMRT** — Intensity-Modulated Radiation Therapy (shapes beams around OARs).
- **Elekta Monaco / TPS** — the hospital's Treatment Planning System (source of the data).
- **`.WC` file** — Elekta Monaco's undocumented plaintext contour format (reverse-engineered here).
- **RTSTRUCT** — DICOM radiotherapy structure-set file (the standard contour format).
- **HU (Hounsfield Unit)** — calibrated CT intensity; here `HU = raw_pixel − 8192`. Model space = raw pixel.
- **Dice** — overlap metric (2|A∩B| / (|A|+|B|)); 1.0 = perfect.
- **Tversky (α=0.3, β=0.7)** — asymmetric Dice-like metric penalising false-negatives (missed tissue) more.
- **HD95** — 95th-percentile Hausdorff Distance (mm); worst-case boundary error, lower is better.
- **Surface Dice @3mm** — fraction of surface within 3 mm tolerance; clinically-acceptable boundary agreement.
- **nnU-Net** — self-configuring SOTA medical-segmentation framework (auto-picks spacing/patch/arch/aug).
- **TTA** — Test-Time Augmentation (nnU-Net's default mirroring at inference — disabled here for clean L/R).
- **Fold / 5-fold ensemble** — nnU-Net's internal 5-fold CV; averaging all 5 folds' predictions = ensemble.
- **PARxxxx** — anonymised nnU-Net case id; `case_mapping.json` maps it back to the real patient id.
- **PBS** — Portable Batch System, the HPC job scheduler; jobs submitted via `.sh` scripts.
- **AMP** — Automatic Mixed Precision (autocast + GradScaler), used to fit/speed up training.
- **The annotation gap** — clinicians deliberately skip the healthy-side parotid on one-sided tumours, so GT
  is intentionally partial → reported Dice underestimates strong models; multi-organ loss must mask absences.
- **The L/R mirror bug** — nnU-Net's default mirror aug made L/R parotids interchangeable (Dice 0.50/HD95 65);
  fixed by the noMirror trainer (Dice 0.82/HD95 5.2). The project's signature debugging result.

---

*End of master reference (Sections 1–19 are the frozen historical brain). Sections below are the LIVING LOG
for the ongoing ablation study — appended by the executing agents. If anything in 1–19 conflicts with a code
file, the code file wins.*

---

## 20. ABLATION STUDY — LIVING LOG (appended during execution)

> This section is maintained by the Claude Code agents executing the ablation study defined in
> `ablation_study/ABLATION_PLAN.md` (governed by `ablation_study/AGENT_INSTRUCTIONS.md`). Sections 1–19
> above are frozen; the study only **appends** here. **Track A appends under §20.A only; Track B appends
> under §20.B only.** The unified synthesis table (§20.C) is written last, once both tracks are done.
>
> Purpose recap: disentangle the 0.62 → 0.82 improvement across its confounded axes — dimensionality (2D vs
> 3D), preprocessing, label quality (the annotation gap), architecture, transformer pretraining, and
> ensembling. Every model is scored on the SAME locked 43-case test set via `pipeline/eval_testset.py`.

### 20.0  Results table (all on the LOCKED TEST set — agents keep rows updated)
| Model / condition | Exp | 3D Dice | Tversky | HD95 (mm) | Surf-Dice | Isolates |
|---|---|---|---|---|---|---|
| nnU-Net 3d_fullres, 1 fold (reference) | — | 0.8187 | 0.8166 | 5.25 | 0.8965 | (existing) |
| nnU-Net 5-fold ensemble (reference) | E5 | 0.8202 | 0.8174 | 5.24 | 0.8990 | ensembling — delta vs 1 fold **n.s.** (p=0.35) |
| **nnU-Net 2d, 1 fold** | **E1** | **0.8117** | 0.8085 | 5.52 | 0.8943 | 2D vs 3D → **3D worth only +0.007** |
| **nnU-Net 3d, GAPPED labels, constant-N 208, 1 fold** | **E2b** | **0.6899** | 0.6680 | 9.63 | 0.7649 | annotation gap, **isolated** → **costs −0.129** |
| **nnU-Net 3d, dirty labels (430), 1 fold** | **E2** | **0.7726** | 0.7644 | 5.45 | 0.8400 | annotation gap, realistic → **net −0.046** |
| U-Net (from scratch, 2D) on test | P0 | 0.7390 | 0.7315 | 4.25* | 0.8702 | baseline on test (*HD95 iso, not comparable to aniso rows) |
| Attention U-Net (scratch, 2D) on test | P0 | 0.7434 | 0.7387 | 3.57* | 0.8827 | baseline on test (best from-scratch) |
| TransUNet (scratch, 2D) on test | P0 | 0.7313 | 0.7307 | 5.60* | 0.8594 | baseline on test |
| Swin-UNet (scratch, 2D) on test | P0 | 0.7156 | 0.7082 | 8.34* | 0.8794 | baseline on test (worst HD95 despite mid Dice → A1) |
| **TransUNet (ImageNet-pretrained, 2D)** | **E3** | **0.7373** | 0.7325 | 3.78* | 0.8759 | pretraining → **+0.006 Dice, −1.82 mm HD95** |
| **Swin-UNet (ImageNet-pretrained, 2D)** | **E3** | **0.7185** | 0.7122 | 6.70* | 0.8679 | pretraining → **+0.003 Dice, −1.64 mm HD95** ⚠️arch caveat |
| Hand-built 3D U-Net (nnU-Net preproc) | E4 | 0.7681 | 0.7746 | 26.76 | 0.8383 | raw; plain 3D recovers most of the gap |
| Hand-built 3D U-Net + largest-CC postproc | E4 | 0.7750 | 0.7788 | 6.05 | 0.8539 | postproc collapses HD95 to ≈nnU-Net; residual −0.044 Dice = training machinery |
| **Custom 3D U-Net, masked loss (430), +CC** | **E6** | **0.7589** | 0.7746 | 9.19 | 0.8403 | annotation-gap FIX (mask); beats dirty-430 (+0.029) but **< clean-208 (0.775) & nnU-Net** — §21 prediction refuted |
| Custom 3D U-Net, dirty labels (430), +CC | E6 | 0.7296 | 0.7267 | 9.61 | 0.7999 | penalise arm (custom U-Net); diverged in training |
| **Per-side specialist experts (L320 + R318), combined** | **E7** | **0.8099** | 0.8096 | 5.85 | 0.8896 | annotation-gap "decompose" arm; competitive but **does not beat nnU-Net 0.8187** on both-parotid |

**Single-side test set (58 held-out one-sided patients, E7 — separate from the 43 both-parotid set; annotated
at-risk gland, Dice+CC):** E7 experts **0.8552** > nnU-Net clean-208 **0.8523** > E4 0.8011 ≈ E6 masked 0.8010
> E6 dirty 0.7436. **The at-risk gland scores ~0.85 — higher than the both-side average — so single-side is
not a failure mode; real-world performance is good.** Four-way annotation-gap treatment (both-parotid set):
**discard 0.8187 (best) > decompose 0.8099 > mask 0.7589 > penalise (dirty, worst)** → no single-side-data
strategy beats using the clean both-parotid set (**data quality > quantity**). Full detail: §20.E (E6), §20.F (E7).

### 20.A  Track A log (nnU-Net: E1, E2, E5)
_(Track A appends dated entries here — command, headline metrics, 1–3 sentence interpretation.)_

**2026-07-16 — E5 (ensembling delta) ✅ done.**
`python3 ablation_study/E5_ensembling/analyze_ensembling.py` — analysis only, no training/inference (both
models were already scored on the locked test set; §15.2). Added a **paired per-case** comparison over the 43
test cases, because a +0.0015 Dice delta cannot be interpreted from two aggregate means.
**Headline:** single fold 0.8187 / HD95 5.25 vs 5-fold ensemble 0.8202 / HD95 5.24.
**Paired result:** mean delta **+0.0016 Dice** (median +0.0066, range −0.055…+0.038), ensemble better on
**26/43** and *worse on 17/43*; **Wilcoxon p = 0.346** (HD95: −0.02 mm, p = 0.264).
**Interpretation:** the 5-fold ensemble buys **nothing measurable** here — the gain is statistically
indistinguishable from zero and is ~2% of the per-case Dice spread (std 0.078); it reshuffles cases rather
than uniformly improving them. This **removes ensembling from contention** as an explanation for the
0.62→0.82 jump (~0.15% of the ~20-point gain), narrowing attribution to dimensionality / preprocessing /
label quality. Practically, the single fold is the better deliverable: 5× the compute for a noise-level gain.
(Expected, in hindsight: the 5 folds share one 208-case pool and an identical config, so they are highly
correlated.) Caveat: n=43 → "no detectable effect at this sample size", not proof of exactly zero.
Full write-up: `ablation_study/E5_ensembling/RESULT.md`; per-case data: `paired_per_case.csv`.

**2026-07-16 — Two corrections to frozen sections (§17.1, §18) found while setting up.** Both are stale;
§1–19 are frozen, so the corrections are recorded here:
(a) **The nnU-Net checkpoints and 5-fold results ARE present locally** under `Parotid-Project/Results/`
(`nnunet_results_Dataset002_noMirror/`, `nnunet_results_Dataset002_5fold/` with all 5 folds + logs) —
§17.1 says they are "NOT in this local folder in full".
(b) **The full nnU-Net raw datasets ARE present locally** at `Parotid-Project/Datasets/` —
`Dataset001_Parotid` (6.6 GB, 208 train + 44 test) and `Dataset002_Parotid` (6.5 GB, **208 train + 43 test**,
verified against §14.3: 251 kept, PAR0240 QC-dropped, exactly 1 L/R flip). §18 describes this folder as
holding only "nnU-Net `dataset.json` + `case_mapping*.json`" — it holds the complete
`imagesTr/labelsTr/imagesTs/labelsTs` volumes. **No rebuild from npz is needed for E1/E2.**
(c) Compute: the local Mac has **no CUDA GPU** (MPS only, 8 GB unified memory, 8 CPUs), so E1/E2/E2b training
must run on a rented GPU. Owner is renting a RunPod pod; the full copy-pasteable sheet (upload, trainer
registration, preprocess, train, predict, eval, sanity checks) is `ablation_study/RUNPOD_COMMANDS_TrackA.md`.

**2026-07-16 — E2 (annotation gap) 🟡 datasets built & verified; GPU run pending.**
`python3 ablation_study/E2_annotation_gap/scan_parotid_presence.py` (5.9 s, CPU, read-only over
`ML_Dataset_Final`) — replicates `build_volumes`' CT-group selection to count what the dirty builder would
actually produce.
**Measured the annotation gap in the train/val pool (n=667):** both=**208**, only_R=**110**, only_L=**112**,
none=237 → clean pool **208** vs dirty pool **430**, i.e. the dirty set adds **222 single-side patients** and
**51.6% (222/430) of parotid-bearing train/val patients are contoured on one side only.** These counts
validate exactly against Dataset002's 208 (§14.3) and against §7's split table (98+12=110 ✓, 99+13=112 ✓,
200+37=237 ✓).
**Validity check (important):** `fix_labels_qc.relabel_geometric()` returns early on single-gland cases, so
dirty single-side cases keep their source L/R naming. Verified safe — **222/222** single-side glands sit on
the anatomically expected side of the midline, with centroids (only_R 202.2 / only_L 306.7) matching the
both-annotated reference (R 202.0 / L 306.4). So E2 isolates the annotation gap and **not** L/R noise.
**Bonus:** the same scan independently reproduces §15.5's key claim from the raw npz — **207/208** of
both-annotated cases agree with the geometric rule, i.e. exactly **1 disagreement**, matching "the geometric
relabel corrected only 1 of 251 cases". The GT really is geometrically consistent; the flips were the model's.
**⚠️ Design issue found → fixed.** E2 as specified changes **two** variables — label condition *and*
training-set size (208 → 430, +107%) — which violates the plan's one-variable rule and would make a null
result ambiguous ("gap is free" vs "gap cost was offset by 2× data"). **Owner approved adding E2b**, a
constant-N arm. Three-arm design: **clean-208 (0.8187) / E2b gapped-208 / E2 dirty-430**, where
*clean − E2b* = the pure cost of the gap, *E2 − E2b* = what the 222 extra patients buy back, and
*clean − E2* = the realistic net effect the Phase-1 baselines actually experienced. Hypothesis: E2 lands
below 0.8187 but is a **lower bound** on the gap's cost; E2b is the number that measures it.

**Datasets built locally (CPU) and verified — nothing on disk modified (sources read-only; images
hardlinked, so the originals cannot be mutated; Dataset002 re-verified intact at 208/208/43/43 after):**
- **`Dataset003_ParotidDirty`** (E2) — `ablation_study/E2_annotation_gap/nnUNet_raw/`, 12 GB, ~4 min.
  **numTraining 430 = 208 both + 110 only_R + 112 only_L** (exactly as the scan predicted), 237 skipped
  (no parotid), 43 test cloned unchanged. Built as *exactly* **clean ∪ single-side**: Dataset002's 208
  training cases are cloned byte-for-byte and only the 222 single-side patients are reconstructed, so the
  both-annotated subset cannot drift from the baseline's. QC threshold 18,622 vox (3× Dataset002 median
  6,208; §14.3's 18,681 was computed over Dataset001 incl. PAR0240 + 44 test — 0.3% apart, dropped nothing).
- **`Dataset004_ParotidGapped`** (E2b) — 30 MB (images hardlinked), ~30 s. **numTraining 208 = 101 both +
  53 only_R + 54 only_L**, simulated gap rate **51.4%** vs the real **51.6%**, deleted-side ratio 53:54
  matched to the real 110:112, **seed 42**, per-case selection recorded. Verified: 208/208 images share an
  inode with Dataset002 (source provably untouched), 0 label files share an inode (all new), **107/107**
  gapped cases removed exactly one gland with the other bit-identical, **101/101** ungapped labels
  bit-identical to Dataset002.
Scripts: `make_dirty_dataset.py`, `make_gapped_dataset.py`. Full detail:
`ablation_study/E2_annotation_gap/RESULT.md`.

**2026-07-16 — E1 (2D vs 3D) 🟡 data ready, GPU run pending.** No CUDA locally; measured on this Mac's MPS
device: 247 ms/step for an encoder-only fwd+bwd at batch 4 / 512×512 → scaling to nnU-Net's real `2d` config
(250 epochs × 250 iters, +decoder, batch 12) gives a **floor of ~26 h**, realistically 2–4 days with OOM risk
in 8 GB; `3d_fullres` is far heavier and not viable. No downscaled substitute was run — it would not be
comparable to the 0.8187 baseline and would answer a different question. E1 needs **no new dataset** (it
trains the `2d` config on the existing clean Dataset002). Exact commands, expected cost (~1–2 h / ~$1–2 on a
4090) and a pre-registered hypothesis (**`2d` lands ~0.72–0.78**, i.e. above the 0.62 Phase-1 ceiling,
because Phase-1 was *also* 2D and plateaued regardless of capacity → the ceiling was the pipeline, not
dimensionality) are in `ablation_study/E1_2d_vs_3d/RESULT.md`.

**2026-07-16 — E1 (2D vs 3D) ✅ DONE. 3D buys almost nothing: +0.007 Dice.**
`nnUNetv2_train 2 2d 0 -tr nnUNetTrainer_250epochs_noMirror --npz` → predict `--disable_tta` → `eval_testset.py`.
1× L40S, **26.4 s/epoch × 250 ≈ 110 min**, finished 19:11.
**Test set (the locked 43): Dice 0.8117 | Tversky 0.8085 | HD95 5.52 mm | Surf-Dice 0.8943**, vs the
`3d_fullres` single-fold baseline 0.8187 / 0.8166 / 5.25 / 0.8965 → **Δ = −0.0070 Dice, +0.27 mm HD95,
−0.0022 Surf-Dice.** Everything except dimensionality was identical (Dataset002, noMirror trainer, fold 0,
same test set). The pre-registered hypothesis said 2d would land 0.72–0.78; **it beat that at 0.8117.**
**Interpretation:** the 2D→3D axis is worth **~0.7 of the ~20 Dice points** in the 0.62→0.82 headline (~3.5%),
and Surface-Dice is effectively identical, so 3D does not even buy boundary quality here. The Phase-1 ceiling
was **not** dimensionality — which vindicates and extends §13.4's diagnosis that the ceiling came from the
data/preprocessing rather than the model. Practically: a 2D model is ~25% cheaper per epoch and gives up
0.007 Dice. Internal fold-0 pseudo-Dice [0.8328, 0.8490] — no mirror bug. `ablation_study/E1_2d_vs_3d/RESULT.md`.

**2026-07-16 — E2b (annotation gap, CONSTANT-N isolation) ✅ DONE. The gap costs 12.9 Dice points.**
`CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror --npz`.
1× L40S, **35.26 s/epoch × 250 ≈ 147 min**, finished 19:48.
**Test set (the locked 43, CLEAN labels): Dice 0.6899 | Tversky 0.6680 | HD95 9.63 mm | Surf-Dice 0.7649**,
vs clean-208 baseline 0.8187 / 0.8166 / 5.25 / 0.8965 → **Δ = −0.1288 Dice, +4.38 mm HD95, −0.1316 Surf-Dice.**
Same 208 patients, same images, same plans (verified spacing [3.0,0.977,0.977] / patch [48,224,192] / batch 2),
same fold, seed 42 — **the labels are the only difference**, so this is a true one-variable measurement.
Tversky (α=0.3/β=0.7, false-negative weighted) drops *further* than Dice — the exact signature expected when
a model is taught to omit glands.
**Note on E2b's low internal pseudo-Dice [0.6595, 0.7606]:** expected, not a red flag — nnU-Net scores each
model against *its own* val labels, and E2b's val labels are gapped too, so it is penalised as a false
positive for correctly predicting an un-contoured gland. The gap corrupts the *metric*; only the eval above,
against the clean test labels, is meaningful. `ablation_study/E2_annotation_gap/RESULT.md`.

**2026-07-16 — E2 (annotation gap, REALISTIC dirty-430 arm) ✅ DONE. Finishes the decomposition.**
`CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror --npz`.
1× L40S, **35.25 s/epoch** × 250 ≈ **147 min**, finished 20:26. Plans verified identical to D002.
**Test set (locked 43, CLEAN labels): Dice 0.7726 | Tversky 0.7644 | HD95 5.45 mm | Surf-Dice 0.8400 |
R 0.7891 / L 0.7560.**

**THE THREE-WAY DECOMPOSITION (the point of adding E2b):**
| Comparison | Δ Dice | Meaning |
|---|---|---|
| clean(0.8187) − E2b(0.6899) | **−0.1288** | the gap's **PURE** cost at fixed N/patients/images/config |
| E2(0.7726) − E2b(0.6899) | **+0.0827** | what the 222 extra single-side patients **buy back** (64%) |
| clean(0.8187) − E2(0.7726) | **−0.0461** | the gap's **NET** cost in the real historical condition |

**Both arms were necessary.** E2 alone → "the gap costs 4.6 pts" (understates it; 2× data masked over half
the damage). E2b alone → "the gap costs 12.9 pts" (true at fixed N, but overstates its historical role). The
confound flagged during prep was real and **larger (0.083) than the effect E2 alone would have reported**.

**⚠️ CORRECTION — retracting the E2b-only claim above.** The E2b entry (written before E2 finished) said the
gap explains **~65%** of the 0.62→0.82 gain. **That is retracted.** It extrapolated a constant-N number to a
historical comparison that did not hold N constant: Phase-1 trained on ~383 parotid-bearing patients (§7:
train 583 → both 186 + only-R 98 + only-L 99), so its label condition matches **E2 (430)**, not E2b (208).

**CORRECTED ATTRIBUTION of the ~0.199 Dice in the 0.62 → 0.82 headline:**
| Axis | Contribution | Share | Source |
|---|---|---|---|
| **Preprocessing** (resample + CTNormalization + augmentation) | **≈ +0.145** | **~73%** | remainder (E2 vs Phase-1, minus 3D) |
| Label quality (as suffered) | ≈ +0.046 | ~23% | clean − E2 |
| Dimensionality 2D→3D | +0.007 | ~3.5% | E1 |
| Ensembling 1→5 folds | +0.0015 (n.s.) | ~0.8% | E5 |

**PREPROCESSING WAS THE DRIVER ALL ALONG** — this proves §13.4's suspicion that the 0.62 ceiling came from
the data/preprocessing rather than model capacity. Not dimensionality, not ensembling, and only ~23% labels.

**Actionable:** partially-labelled patients are **worth including** — they recover 64% of the damage and
restore boundary quality almost fully (HD95 9.63 → 5.45 ≈ the 5.25 baseline) — but still leave 4.6 pts on the
table. **This is the strongest case yet for `pipeline/masked_loss.py` (§10.2, still unused):** masking
un-annotated organs out of the loss should capture all 430 patients' data benefit *without* the 12.9-pt
penalty, plausibly beating 0.8187. A concrete, motivated next experiment.
**Supporting signals:** Tversky (β-weighted to punish false negatives) falls further than Dice in both arms —
the signature of a model taught to *omit* glands. R/L asymmetry widens monotonically with label damage:
clean 0.018 → E1 0.031 → E2 0.033 → E2b 0.044. `ablation_study/E2_annotation_gap/RESULT.md`.

### ✅ TRACK A COMPLETE (2026-07-16). E5 ✅ · E1 ✅ · E2 ✅ · E2b ✅ (bonus arm).
All four scored on the same locked 43-case test set via `eval_testset.py`; all plans verified identical to
the Dataset002 reference; `--disable_tta` and the noMirror trainer throughout; no run showed the 0.4538
mirror-bug signature. Total ≈ $10 on a 3× L40S RunPod box. Artifacts in `ablation_study/_pod_results/`.

**S1 is still BLOCKED — do not run it.** Track B (E3/E4/A1/P0) is not done. Two things S1 must handle:
1. **The 0.62 is a Phase-1 *validation* number** while every number above is *test*, so the preprocessing
   remainder (+0.145) is approximate. **Track B's P0** puts the four Phase-1 checkpoints on the locked test
   set and will firm up the arithmetic. The *ordering* of the axes will not change.
2. **Preprocessing's share is a remainder, not a measurement** — it is what is left after subtracting the
   measured axes. Track B's **E4** (hand-built 3D U-Net on nnU-Net-preprocessed data) is the run that tests
   it directly: if E4 lands near 0.8187, architecture is irrelevant and the pipeline claim is confirmed.

**How Track A gets finished (the GPU work is queued, not abandoned).** All three GPU runs are automated in a
single resumable script, `ablation_study/pod_run_all_trackA.sh` (preflight + case-count assertions → trainer
registration *with a resolve check* → preprocess ×3 → train ×3 (fold 0, `--npz`, noMirror) → predict ×3
(`--disable_tta`, asserts 43 outputs) → eval ×3 against the clean Dataset002 test labels → summary +
`run_manifest.txt` recording GPU/wall-time). Run sheet: `ablation_study/RUNPOD_COMMANDS_TrackA.md`. When the
owner returns the results (`ablation_study/_pod_results`), the agent fills in E1/E2/E2b RESULT.md, the status
rows, the §20.0 table and §20.A, then marks **Track A complete**. Nothing in Track A is being dropped for
lack of a GPU — it is staged to run in one pod session.
What E5 already establishes for the synthesis: ensembling is *out* as an explanation of 0.62 → 0.82, leaving
{3D, preprocessing, labels}, which E1 (3D axis), E2b (label axis, isolated) and E2 (label axis, realistic)
are built to measure; preprocessing is then the remainder.

### 20.B  Track B log (custom PyTorch: P0, E3, E4, A1)
_(Track B appends dated entries here.)_

**2026-07-16 — Track B session: all NON-GPU work done; heavy compute staged for the pod.**
Local Mac has no CUDA (MPS/8 GB); per the owner, all training-shaped work and the P0 inference
(measured ~2-3 h on MPS) are deferred to a rented GPU. Everything below is code + configs +
command sheet, **validated locally without training**. Runbook: `ablation_study/RUNPOD_COMMANDS_TrackB.md`.

**2026-07-17 — P0 (Phase-1 checkpoints on the locked test set) ✅ DONE (RunPod L40S).**
All four from-scratch checkpoints scored on the locked 43-case test set with the Phase-1 3D
volumetric protocol (`evaluate_phase1_on_test.py`, `--device cuda`), reading the Dataset002 NIfTI so
Dice sits on the same voxel grid as nnU-Net's 0.8187. Results (n=86 sides/model): **Attention U-Net
0.7434** (HD95 3.57, SurfD 0.8827) / **U-Net 0.7390** (4.25, 0.8702) / **TransUNet 0.7313** (5.60,
0.8594) / **Swin-UNet 0.7156** (8.34, 0.8794). Interpretation: on the clean test cases all four cluster
**0.72–0.74** — ~0.11 above their 0.62 *val* numbers (§13.2) and ~0.08 below nnU-Net; the whole
31M→102M range is within 0.028 Dice, so **architecture barely moves the from-scratch 2D ceiling** —
consistent with Track A's "preprocessing is the driver". Swin is the boundary outlier (worst HD95 by
far, yet *fewest* one-sided misses 2/86 vs 7–8 — its HD95 tail is stray islands, e.g. 135/96/89 mm per
case, not missed glands → sets up A1). Every model scores R>L with mean<<median (a few catastrophic
one-sided cases drag the mean, the §15 L/R/partial-label mode). HD95/SurfD are Phase-1 **isotropic**
(caveat vs nnU-Net's anisotropic); Dice/Tversky are comparable. Operational notes: pip deps are wiped
on every pod restart (reinstall each time); the first Dataset002 upload dragged macOS `._*` files that
broke nibabel globs (deleted with `find -name '._*' -delete`); TransUNet's checkpoint was truncated in
transit (`1211352136` vs `1230961638`) and re-uploaded. Files: `P0.../eval.csv`, `per_case.csv`,
`preds/` (172). **P0 ✅.**

**(superseded) P0 pre-run staging note — 🟡 code ready + validated, GPU-pending.**
`evaluate_phase1_on_test.py` reproduces the frozen Phase-1 3D volumetric protocol
(`phase1_metrics.py` copied behaviourally from `attention_evaluate.py`) and scores the four
`.pth` on the **Dataset002 NIfTI test volumes** — the *exact same 43 cases and GT voxels* as
nnU-Net's 0.8187, so **Dice/Tversky are directly comparable**. Verified the data source: `imagesTs`
is raw pixel (min 0/max ~27k) so `window_and_normalise` applies unchanged, and label 1/2 = R/L by
column centroid. **Caveat carried into the write-up:** P0 HD95/Surf-Dice use the Phase-1 *isotropic
0.977* convention → NOT comparable to nnU-Net's anisotropic HD95 (Dice/Tversky are). Local smoke
check (6 U-Net cases on MPS before moving to GPU): U-Net loads at epoch 4 / val_loss 0.1685; sensible
per-case Dice (~0.85-0.92) **plus two one-sided L/R misses** (PAR0212 L≈0.05, PAR0213 R≈0.00) — the
from-scratch 2D models also show the L/R-confusion mode on a subset. Full numbers pending.

**2026-07-17 — A1 (Swin boundary failure, Q5) ✅ DONE (RunPod, CPU).**
Ran `analyze_swin_failure.py` on P0's 172 saved predictions (Swin + U-Net, 86 glands each). **Both
master hypotheses confirmed, cleanly separated.** Per-gland Swin vs U-Net: connected components 2.84 vs
1.20 (max 9 vs 5), multi-component 75.6% vs 16.3%, stray components 1.85 vs 0.23, stray-voxel fraction
5.03% vs 1.67%, farthest stray island 55.4 mm (max 306) vs 12.2 mm — all four island metrics separate
at **paired Wilcoxon p < 1e-8** (Swin more fragmented on 62/86 glands). **HD95 full→largest-CC-only:
Swin 14.30→10.53 (−3.77 mm) vs U-Net 8.53→8.49 (−0.04 mm)** — i.e. **H2 (underfit → stray islands) is
the dominant driver**: most of Swin's HD95 inflation is far-flung false-positive islands nnU-Net-style
CC cleanup removes. **H1 (coarse 128→512 upsampling) is the residual floor**: even after island removal
Swin's largest-CC HD95 (10.53) stays ~2 mm above U-Net's (8.49), a systematic penalty on the main
gland's boundary (`swin_unet.py:402-404`). This answers Q5 — Swin's boundary metrics are bad for two
compounding reasons (coarse-boundary floor + stray-island tail), consistent with P0 (Swin had the worst
HD95 yet fewest one-sided misses: it finds glands, messily). Fixed a perf bug en route (added bbox-crop
to the distance transforms; the un-cropped version ran >60 min, the cropped one ~2–3 min). Figures:
`montage_islands.png` (PAR0231 z=65), `montage_coarse.png`. Files: `per_gland.csv`, `summary.json`.
**A1 ✅. Track B is now P0 ✅ / E4 ✅ / A1 ✅ — only E3 remains.**

**(superseded) A1 pre-run staging note — 🟡 code ready + validated on synthetic.**
`analyze_swin_failure.py` reuses P0's saved prediction volumes (`--save-preds`) to test both master
hypotheses: **H1 coarse output** (confirmed structurally in `swin_unet.py:402-404` — logits at
128×128 then bilinear→512) and **H2 underfit/stray-islands** (confirmed `training_history_swinunet.json`
floors at val_loss 0.2101@ep13 vs CNNs ~0.16). It counts 3D connected components per gland,
stray-island mass/distance, and **HD95 full vs largest-component-only** (Swin vs U-Net, paired
Wilcoxon), plus `make_montage.py` figures. Synthetic-data validation confirms the metrics separate
the two signatures (stray comps, ~47 mm island). Pure CPU/seconds once P0 preds exist; no GPU of its own.

**E3 (transformer pretraining, Q3) — 🟡 all non-GPU work done incl. real ImageNet loads.**
`models_pretrained.py` + `validate_e3.py`: **TransUNet-pretrained** loads torchvision ResNet-50
IMAGENET1K_V2 into the frozen encoder — **258/258 tensors, conv1 stem 3→1 averaged** (clean,
same-architecture). **Swin-pretrained** = timm `swin_tiny_patch4_window7_224` encoder (ImageNet,
in_chans=1, img_size=512; stages [96,192,384,768]) + the frozen Swin decoder blocks. Both pass
forward+backward at 512²; the real pretrained fetch works (internet verified). `train_e3.py` /
`evaluate_e3.py` reproduce the Phase-1 2D protocol (reuse frozen `hdf5_dataloader`+`loss_function`;
eval reuses `phase1_metrics` on the same 43 cases as P0). **⚠️ Design flaw flagged (golden rule #6):**
a true same-architecture pretrained Swin is impossible — the frozen Swin uses window=8/512
(rel-pos-bias 225, depths [2,2,2]) vs ImageNet Swin-T's window=7/224 (169, [2,2,6,2]); the weights
are un-loadable, so the Swin arm changes the encoder → **mild architecture confound on the Swin
comparison only** (TransUNet arm is clean). Raised with owner in `E3.../RESULT.md`. Data: needs a
~30 GB train+val 2D `dataset.h5` — `build_compact_h5.py` written + validated (drops test patients &
non-parotid organs; frozen dataloader reads it). Upload is the ~$8-10 item (Track A learning #6).

**2026-07-17 — E4 (custom 3D U-Net, Q4) ✅ DONE (RunPod L40S, ~50 min).**
Plain hand-built 3D U-Net (16.5 M, features [32,64,128,256,320], plain isotropic pooling — a
deliberately vanilla, *different* implementation) trained on nnU-Net-preprocessed Dataset002 (CTNorm
constants from `plans.json`, spacing already [3,0.977,0.977] so no resample), fold-0 (166/42),
early-stopped epoch 83 (best val 0.1222), Gaussian sliding-window inference, scored with the same
`eval_testset.py` as nnU-Net (true anisotropic). **Result: Dice 0.7681, Tversky 0.7746, HD95 26.76 mm,
Surf-Dice 0.8383 (R 0.799 / L 0.737) vs nnU-Net 3d_fullres 1-fold 0.8187 / 5.25 / 0.8965 → −0.0506
Dice.** Finding: a plain 3D U-Net on the correct pipeline reaches 0.77 — **above the from-scratch 2D
ceiling (0.72–0.74 on test, P0) but ~0.05 under nnU-Net**, so **architecture is largely (not entirely)
irrelevant once the pipeline is right**; the residual traces to nnU-Net's training machinery (deep
supervision, heavy aug, anisotropic strides) and postprocessing, not the core architecture. The
loud signal is **HD95 26.76 vs 5.25**: with Dice fine, the blow-up is **stray false-positive islands**
(this net has no largest-CC postprocessing that nnU-Net applies) — the same failure mode A1 quantifies
for Swin and that §15.3's cheap CC/L-R-split postproc fixes. **Refines Track A's attribution:** P0
shows Phase-1 *test* is ~0.74 (not the 0.62 *val*), so much of the headline 0.62→0.82 was val→test
measurement; the true from-scratch-2D→nnU-Net gap on test is ~0.08, split ~1/3 plain-3D-pipeline (E4) /
~2/3 nnU-Net machinery+postproc. Ordering (preprocessing/3D ≫ architecture) unchanged; S1 must carry
the val-vs-test caveat. **Largest-CC postproc (RAN):** `postproc_largestcc.py` (keep largest component per class) + re-eval
gave **HD95 26.76 → 6.05 mm (≈ nnU-Net 5.25), Dice 0.7681 → 0.7750 (up), Surf-Dice 0.8383 → 0.8539** —
the whole HD95 gap was stray islands, removed by one line of postprocessing, **not architecture**. Net
statement: a plain hand-built 3D U-Net on nnU-Net's pipeline + trivial postproc matches nnU-Net on
boundaries and lands within **0.044 Dice**; the residual is nnU-Net's training machinery (deep
supervision, aug), not the model. Files: `E4.../eval.csv` (both rows), `ckpt_unet3d/`, `pred_test/`,
`pred_test_cc/`. **E4 ✅.**

**(superseded) E4 pre-run staging note — 🟡 all non-GPU work done, validated end-to-end.**
Plain hand-built 3D U-Net (`unet3d.py`, 16.5 M, features [32,64,128,256,320], InstanceNorm,
2-ch sigmoid — a deliberately vanilla, *different* implementation). `preprocess_e4.py` reproduces
nnU-Net 3d_fullres preprocessing **exactly**: Dataset002 NIfTI are already at [3.0,0.977,0.977] (no
resample) and **CTNormalization constants are read from the run's `plans.json`** (clip [8087,8394],
(x−8198.65)/57.49). `dataloader_e4.py` does patch [48,224,192] + 0.33 foreground oversampling and
**reproduces nnU-Net's default fold-0 split (verified 166/42, matching §14.6)**. `predict_e4.py`
does Gaussian sliding-window → labelmaps in labelsTs geometry, scored by the **same
`pipeline/eval_testset.py`** → **all four E4 metrics directly comparable to 0.8187** (no HD95 caveat,
unlike P0/E3). `validate_e4.py` ran the whole chain on real cases + small tensors. Upload cheap
(6.5 GB Dataset002, shared with P0).

**Track B status (2026-07-16): P0/A1/E3/E4 all 🟡 — every non-GPU part (code, dataloaders, configs,
preprocessing replicas, validations, command sheet) is complete and locally validated; awaiting the
pod (Track B gets GPU time after Track A's runs finish). Track B is NOT complete — do NOT run S1**
(also blocked because Track A's E1/E2 are still 🟡). GPU order when the pod is free: **P0 (+save preds)
→ A1 → E4 → E3** (P0/A1/E4 are cheap-upload; E3 last, pending the big upload decision).

**2026-07-17 — E3 (transformer pretraining, Q3) ✅ DONE. ANSWER: NO — pretraining does not fix them.**
`train_e3.py --model {transunet_pretrained,swin_pretrained} --samples-per-epoch 12000 --epochs 40 --patience 8`
then `evaluate_e3.py` (P0's `phase1_metrics`, isotropic, 86 sides). 1× L40S, seed 42, AMP.
TransUNet: 258/258 torchvision ResNet-50 IMAGENET1K_V2 loaded, conv1 stem 3→1 averaged, 102.5M params
(**identical to P0's from-scratch TransUNet — only the initialisation differs**); early stop @23, best val
0.1810, 94.3 min. Swin: timm `swin_tiny_patch4_window7_224` encoder, 48.3M; early stop @22, best val 0.1703,
55.5 min. Data: `trainval.h5` 10.62 GB, 667/667 patients, 34,415 slices (verified byte-exact after upload).

**Test set (locked 43, Phase-1 isotropic protocol):**
| Model | Dice | Tversky | HD95 | Surf-Dice | Δ vs from-scratch (P0) |
|---|---|---|---|---|---|
| TransUNet pretrained | **0.7373** | 0.7325 | **3.78** | 0.8759 | **+0.0060 Dice, −1.82 mm** |
| Swin pretrained ⚠️ | **0.7185** | 0.7122 | **6.70** | 0.8679 | **+0.0029 Dice, −1.64 mm** |

**Interpretation.** (1) Pretraining does **essentially nothing for overlap** — +0.006 / +0.003 sit inside
per-case noise (per-case Dice std ≈ 0.078). (2) It **does clearly help boundaries** — HD95 −1.8 mm on both
arms, consistently; pretrained features localise gland *edges* better, they just don't find more gland.
(3) **The ranking is unchanged**: the best pretrained transformer (TransUNet 0.7373) **still loses to the
from-scratch Attention U-Net (0.7434)**. ⇒ The transformers' Phase-1 weakness was **NOT** the from-scratch
handicap. This **half-refutes §13.4**: the data *is* too little for a ViT, but pretraining does not rescue it
— the **dataset**, not the initialisation, is the ceiling. (4) Mechanism: TransUNet's pretrained run drove
**train loss to 0.0545**, a third of its own val (0.1810) and below the from-scratch model's best-ever val
(0.1692) — it memorised the 6,934 parotid slices without generalising.

**⇒ ALL SIX 2D models now span just 0.0278 Dice (0.7156–0.7434) — architecture AND initialisation combined.**
Against preprocessing ≈ +0.145 and labels ≈ +0.129 (latent), model choices are ~5× less important. Sixth
independent confirmation of the study's thesis.

**⚠️ Val loss and test Dice disagree on BOTH arms — do not cite the val numbers.** TransUNet val
0.1692→0.1810 (worse) yet test +0.006 (better); Swin val 0.2101→0.1703 (much better) yet test +0.003 (flat).
Causes: the capped h5 (empty slices ≤40/patient) changes val batch composition, and val loss is a 2D
slice-level quantity while the reported Dice is 3D volumetric (§13.1's warning). **Only the test Dice is
authoritative.**
**⚠️ Caveats:** the Swin arm carries a documented architecture confound (a true same-arch pretrained Swin is
impossible — frozen window=8/512, bias table 225 vs ImageNet Swin-T window=7/224, table 169), so it is
48.3M params vs P0's 27M; **the TransUNet arm is clean and carries the headline claim**. Budget deviations
(empty-slice cap; 12,000 samples/epoch vs Phase-1's full pass) affect absolute values only — the
pretrained-vs-scratch **delta** applies the same cap to both sides. E3/P0 HD95 are **isotropic** and not
comparable to nnU-Net/E4's anisotropic numbers (§12.3/§14.5). `ablation_study/E3_transformer_pretrain/RESULT.md`.

### ✅ TRACK B COMPLETE (2026-07-17). P0 ✅ · A1 ✅ · E4 ✅ · E3 ✅.
All scored on the same locked 43-case test set. Artifacts pulled to `ablation_study/_pod_results_B/`
(3 checkpoints verified loading, 172 P0 preds, 43+43 E4 preds, A1 figures, all eval CSVs). Pod terminated.
**Both tracks are now complete → S1 (synthesis) is UNBLOCKED.**

### 20.C  Synthesis ✅ (2026-07-17 — both tracks complete; full text: `ablation_study/SYNTHESIS.md`)

## ⚠️ THE HEADLINE IS WRONG AND MUST BE RETIRED: **~62% of "0.62 → 0.82" was a measurement artifact.**

The 0.62 is a **validation** number; the 0.82 a **test** number. **P0 re-scored the four Phase-1 checkpoints
on the locked 43-case test set — the same weights score +0.1325 higher on test than on val:**

| Model | VAL (§13.2) | TEST (P0) | Δ |
|---|---|---|---|
| U-Net | 0.6209 | 0.7390 | +0.1181 |
| Attention U-Net | 0.6346 | **0.7434** | +0.1088 |
| TransUNet | 0.6332 | 0.7313 | +0.0981 |
| Swin-UNet | 0.5106 | 0.7156 | **+0.2050** |
| mean | 0.5998 | 0.7323 | **+0.1325** |

**The defensible statement is `0.7434 → 0.8187 (+0.0753)` on the same held-out test set.** Never quote
0.62 → 0.82 again. **Why it happened:** Phase-1's val set = 47 parotid-bearing val patients **including
single-side (partial-label) cases**; the locked test set = 43 **both-parotid QC-clean** cases. §16.1 predicted
exactly this — partial GT penalises a model for correctly predicting an un-contoured gland. **The 0.62 was
deflated by the annotation gap on the EVALUATION side.**

## Decomposition of the REAL +0.0753 (each a controlled A/B on the locked test set)

| Axis | Δ Dice | Share | Experiment |
|---|---|---|---|
| **Label quality (net, as suffered)** | **+0.0461** | **~61%** | E2: dirty-430 0.7726 → clean 0.8187 |
| Preprocessing + nnU-Net training recipe | ~+0.0222 | ~29% | remainder (0.7434 → 0.7726, minus 3D) |
| Dimensionality (2D→3D) | +0.0070 | ~9% | E1 |
| Ensembling (1→5 folds) | +0.0015 (**n.s.** p=0.35) | ~2% | E5 |

**⚠️ RETRACTION.** §20.A's interim entry claimed **preprocessing ≈ +0.145 (~73%)** and predicted *"the
ordering of the axes will not change"* once P0 landed. **Both were wrong.** That figure used the 0.62 **val**
baseline before P0 existed; with Phase-1's true **test** score (0.7434) the ordering **inverts — labels lead,
preprocessing is second.** The §20.A entry stands as written (the log is append-only); this is the correction.

## The annotation gap is the thread through the entire project — it hits BOTH sides

| Where | Effect | Exp |
|---|---|---|
| **Training** (latent, constant N) | **−0.1288** Dice, HD95 5.25→9.63 | E2b |
| **Training** (net, 2× data compensating) | −0.0461 Dice | E2 |
| **Evaluation** (deflates val) | **−0.1325** Dice on identical weights | P0 vs §13.2 |
| Prevalence | **222/430 = 51.6%** of parotid-bearing train/val patients | E2 scan |
| Structure | E2−E2b = **+0.0827** → the 222 extra partial patients buy back **64%** of the damage | E2+E2b |

⇒ **`pipeline/masked_loss.py` (§10.2, written, unit-tested, NEVER USED) is the best-motivated next
experiment in the project** — it should capture all 430 patients' data benefit *without* the 12.9-pt penalty,
plausibly **beating 0.8187**. Now a number-backed prediction, not a design intuition.

## Architecture & initialisation are near-irrelevant (5 independent lines)

- **Six 2D models** (4 scratch + 2 pretrained) span **0.0278 Dice** (0.7156–0.7434).
- **E3:** ImageNet pretraining = **+0.006** (TransUNet, 258/258 exact load, identical 102.5M) / **+0.003**
  (Swin) — noise. Best pretrained transformer (0.7373) **still loses to from-scratch Attention U-Net
  (0.7434)**. Pretraining's real benefit is **boundaries** (HD95 −1.8 mm both). ⇒ **half-refutes §13.4**.
- **E4:** hand-built plain 3D U-Net on nnU-Net preprocessing = 0.7681; **+largest-CC postproc → HD95
  26.76→6.05 mm** (≈ nnU-Net's 5.25) at ~constant Dice ⇒ **nnU-Net's boundary edge is postprocessing, not
  architecture**; the −0.0437 residual is its *training machinery*.
- **E1:** 2D vs 3D = +0.007. **§13.4:** 31M→102M params = no gain.

## Swin-UNet was never as bad as §13.2 suggests (P0 + A1)

**P0:** on test Swin = **0.7156**, a **+0.2050** jump from its 0.5106 val — the largest of any model, and only
0.028 behind the best CNN. Its "collapse" was largely a val-set artifact (the annotation gap punishing its
bilateral predictions). **A1:** the *boundary* failure is real — **H2 stray islands dominant** (2.84 vs 1.20
components, 75.6% vs 16.3% multi-component, strays up to 306 mm; removing them drops HD95 3.77 mm vs 0.04 for
U-Net, p=1.3e-11), **H1 coarse 128→512 upsample residual** (Swin-lcc 10.53 still > U-Net 8.49). **Q5 ✅.**

## Answers

**Q1** 3D is only +0.007 (9% of the real gain) — the ceiling was not dimensionality, and mostly not real.
**Q2** The annotation gap is the largest single driver: −0.129 latent / −0.046 net, plus −0.133 on evaluation.
**Q3** No — pretraining doesn't fix the transformers (+0.006/+0.003); they still lose to a from-scratch CNN.
**Q4** Barely — six models span 0.028; a hand-built 3D U-Net + CC postproc matches nnU-Net's HD95.
**Q5** Stray islands (dominant) + coarse upsampling (residual), both p<1e-8.

## Key limitations (§9 of SYNTHESIS.md has all nine)

**n=43, per-case Dice std ≈ 0.078 → any delta below ~0.02 is noise**, which includes E1 (+0.007),
E3 (+0.006/+0.003) and E5 (+0.0015) — reported as "no measurable effect", not zero. Single fold / single seed
(no seed-variance estimate). **HD95 isotropic (P0/E3/A1) vs anisotropic (nnU-Net/E2/E4) — never one column.**
Attribution is **not strictly additive** (E4 shows the training machinery alone, +0.0437, exceeds the +0.0222
"preprocessing" remainder). **The test set is the easy subset** (43 both-parotid QC-clean) — 0.8187 is an
optimistic operating point and nothing here measures single-side/messy-case performance.

### ✅ ABLATION STUDY COMPLETE — E1, E2, E2b, E5 (Track A) · P0, A1, E3, E4 (Track B) · S1.

---

## 21. MASKED PARTIAL-LABEL LOSS — PREDICTION MADE, THEN TESTED IN E6 (result: not confirmed)

> **⚠️ UPDATED 2026-07-17 — this experiment has now been RUN (E6). The prediction below did NOT hold.**
> Masking the un-annotated gland **beats the naive all-data control (+0.029 Dice) and trains far more
> stably**, but on the locked both-parotid test set it reached only **0.759 (+CC)** — *below* the clean-208
> baseline (0.775) and nnU-Net (0.819). So the "should beat 0.8187" prediction is **refuted on this test
> set**. Full result + caveats in **§20.E**; the honest next step is a **single-side evaluation** (this test
> set is all both-parotid and structurally can't reward masking's main benefit). The original prediction is
> preserved below as written (for the record), followed by the finding.

**Original status (pre-E6):** the single best-motivated next experiment in the project — the intervention the
ablation evidence predicted could **beat the current best (0.8187)**.

**The idea (why it should work).** 51.6% of parotid-bearing patients are single-side (the clinician
deliberately skipped the healthy gland — the annotation gap, §16.1). When such a slice is used in training,
the un-contoured side's GT channel is all-background, so an ordinary Dice+BCE loss **punishes the model for
correctly predicting that gland**, teaching it to suppress correct anatomy. The ablation quantified this
cost: **−0.129 latent (E2b) / −0.046 net (E2)**. The masked partial-label loss (`pipeline/masked_loss.py`,
§10.2 — already written and unit-tested, **never used**) computes the loss **only over the organs actually
annotated for each patient**, so the un-annotated side is masked out entirely. This lets training use **all
430 parotid-bearing patients** (keeping the +0.083 data benefit E2 showed) **without** the label penalty.

**Number-backed prediction:** capturing the extra-data gain without the noise cost should land **above the
clean-label reference 0.8187**. This is now a prediction with an evidence trail, not a design hunch.

**Implementation wrinkle (important):** nnU-Net expects complete integer labelmaps and does **not** natively
support masked partial-label loss. So this experiment is run most cleanly in **custom 3D training — reuse
E4's hand-built 3D U-Net (`ablation_study/E4_custom_3d_unet/`, which already trains on nnU-Net-preprocessed
Dataset-style volumes) and swap the loss for `masked_loss.py`** — or by patching nnU-Net's loss module.
Prefer the E4-based route: it's proven to work on this data and keeps the comparison clean.

**Evaluate** on the SAME locked 43-case test set via `pipeline/eval_testset.py`, and compare to 0.8187
(nnU-Net single fold) and E4's 0.7681/0.7750 (same custom U-Net, ordinary loss). Also report on single-side
cases specifically — this is where the fix should most visibly help, and §20 notes the current 43-case test
set is the "easy subset" that measures none of that.

---

## 22. INDEPENDENT VERIFICATION & ARTIFACT LOCATIONS (main-agent audit, 2026-07-17)

The main agent independently re-checked the ablation results against the **raw eval CSVs and per-case files**
(not the writeups). Recorded here so the audit lives in the master doc, not only in the folder.

### 22.1 Verified ✅ (recomputed from raw data, matches SYNTHESIS/§20 exactly)
- **Unified table (all 13 rows)** matches the raw `eval.csv` files to the digit: P0 (U-Net 0.7390 / Attn
  0.7434 / Trans 0.7313 / Swin 0.7156), E1 2d 0.8117, E2 dirty 0.7726, E2b gapped 0.6899, E3 (Trans-pre
  0.7373 / Swin-pre 0.7185), E4 0.7681 raw → 0.7750 CC.
- **Val→test artifact (+0.1325 mean)** — recomputed from `P0/per_case_full.csv`; the same weights genuinely
  score ~+0.13 higher on the clean test set. Real.
- **E5 ensembling n.s.** — recomputed from `E5_ensembling/paired_per_case.csv`: mean delta **+0.0016**,
  ensemble *worse* on **17/43** cases. Statistically zero, as claimed.
- **A1 Swin islands** — confirmed from `A1_swin_failure/summary.json` (2.84 vs 1.20 components/gland; HD95
  drop 3.77 vs 0.04 mm; p≈1.3e-11).
- **Structure:** Dataset003 numTraining=430, Dataset004=208; single-side prevalence **222/430 = 51.6%** exact;
  all 6 checkpoints physically present (2d 354M · dirty 235M · gapped 235M · E3 Swin 559M · E3 Trans 1.2G ·
  E4 3D-UNet 64M).

### 22.2 One correction to fold in — the noise-floor number
§20's limitations say *"per-case Dice std ≈ 0.078 → deltas below ~0.02 are noise."* On recomputation:
- The **single-model per-case Dice dispersion is ≈ 0.17**, not 0.078 (case difficulty varies a lot).
- **0.078 is really a paired-difference std** (the correct quantity for an A/B test, since case difficulty
  cancels in the pairing). Measured paired-delta stds: E5 (same arch, single-vs-ensemble) **0.017**;
  Attn-vs-U-Net **0.047**; genuinely different cross-run pairs plausibly ~0.05–0.08.
- **Net effect on conclusions:** the "deltas < ~0.02 are noise" rule is directionally right, but the honest
  cross-run paired noise floor is closer to **~0.05**. That makes **E2's net −0.046 borderline** (not
  comfortably significant). **The robust claims are unaffected:** E2b latent **−0.129** and the eval-side
  **+0.133** are both far above any noise floor, so *"the annotation gap is the dominant factor"* stands
  firmly. What softens is only the precise *"labels = 61% of the +0.075"* decomposition — report the annotation
  gap's dominance via the latent/eval numbers, and treat the exact percentage split as indicative.

### 22.3 Full artifact map (everything the study produced, all on the Mac)
```
ablation_study/
  ABLATION_PLAN.md · AGENT_INSTRUCTIONS.md · TRACK_A/TRACK_B_*.md · RUNPOD_COMMANDS_*.md
  SYNTHESIS.md                          <- the S1 synthesis (mirrored in §20.C)
  <Eid>/RESULT.md                       <- per-experiment writeups (E1,E2,E5,P0,A1,E4,E3)
  E1_2d_vs_3d/ · E2_annotation_gap/ (incl. nnUNet_raw Dataset003=430 / Dataset004=208) ·
  E5_ensembling/paired_per_case.csv · P0_phase1_on_test/{eval.csv,per_case_full.csv} ·
  A1_swin_failure/ · E4_custom_3d_unet/{eval.csv,RESULT.md} ·
  E3_transformer_pretrain/data/trainval.h5 (10.6 GB, rebuildable via build_compact_h5.py)
  _pod_results/      (830 MB — Track A nnU-Net checkpoints + preds + eval CSVs + run_manifest.txt)
  _pod_results_B/    (1.6 GB — Track B checkpoints + P0 preds + A1 figures + eval CSVs)
POD_UPLOAD_PLAYBOOK.md                   <- Mac→RunPod transfer lessons (repo root)
```
Both pods terminated; nothing left running; no compute pending.

### 20.E  Masked partial-label loss (E6) — ✅ DONE (2026-07-17; §21 prediction NOT confirmed)
_(The E6 agent appends here — task spec in `ablation_study/E6_masked_loss_tasks.md`. Goal: on the custom 3D
U-Net (E4), run three arms — clean-208 [=E4, exists], dirty-430, masked-430 [`masked_loss.py`] — all scored
on the locked 43-case test set, to test whether masking un-annotated glands recovers the annotation-gap cost
(§21's number-backed prediction). Append: commands, three-arm metric table, cost, mechanism verdict, caveats.)_

- 🟡 **Local prep complete (2026-07-17, CPU, $0)** — code, preprocessing, and mask plumbing all staged and
  verified; both GPU runs pending the owner's HPC-vs-RunPod decision. Full record: `ablation_study/E6_masked_loss/RESULT.md`.
  - Built under `E6_masked_loss/` by reusing E4 verbatim (`unet3d.py`/`loss3d.py`/`postproc_largestcc.py` copied)
    plus `preprocess_e6.py` (Dataset003 430 train + Dataset002 43 test, E4's exact norm constants, +per-case mask),
    `dataloader_e6.py` (emits `(img,target,mask)`; hflip swaps the mask with the R/L channels),
    `train_e6.py` (`--loss dirty|masked`), `predict_e6.py`, `sanity_mask.py`.
  - Preprocessed **430 train + 43 test → 2.0 GB compact cache** (upload this, not raw NIfTI). Composition verified:
    **both=208 / only_R=110 / only_L=112 → single-side 222/430 = 51.6%** (exact match to E2).
  - **`sanity_mask.py` passes end-to-end:** masked loss ignores the un-annotated channel (Δ=0.0), ordinary loss
    does not (Δ=0.167); masked==ordinary on both-annotated (Δ=1.6e−6); hflip mask-swap verified. `masked_loss.py`
    unit test passes.
- ✅ **dirty-430** (custom 3D U-Net, ordinary loss, 430 patients) — trained (2× L40S, RunPod). Test Dice
  **0.7111 raw / 0.7296 +CC** (HD95 47.63 → 9.61). **Diverged hard mid-training: val loss collapsed from
  0.203 (best @ ep45) to 0.471 and stayed there through ep75** (verified in `training_history.json`). ⇒ the
  +0.029 masked-over-dirty gap is **partly a training-stability artifact**, not purely the label penalty;
  de-confound with gradient clipping + multiple seeds before citing it as a clean label effect.
- ✅ **masked-430** (custom 3D U-Net, `masked_loss.py`, 430 patients) — trained. Test Dice
  **0.7079 raw / 0.7589 +CC** (HD95 134.69 → 9.19, biggest CC gain of any arm). Trained stably (best @ ep62).

**E6 RESULT (2026-07-17, ~$2, all on the locked 43-case test set, `eval_testset.py`, +CC = largest-CC postproc):**

| Arm | Train | Loss | Dice +CC | vs dirty | vs clean-208 |
|---|---|---|---|---|---|
| clean-208 (=E4) | 208 both | Dice+BCE | **0.7750** | — | — |
| **masked-430** | 430 | Masked | **0.7589** | **+0.029** | −0.016 |
| dirty-430 | 430 | Dice+BCE | 0.7296 | — | −0.045 |
| _ref:_ nnU-Net 1 fold | 208 | — | 0.8187 | | |

**Verdict — mechanism partially confirmed, strong prediction (§21) NOT confirmed.**
- **masked-430 > dirty-430 (+0.029 CC): direction holds** — masking the un-annotated gland beats penalising
  it on the same 430 patients. Masked model has good cores but stray islands (raw HD95 134→9 after CC).
- **masked-430 (0.7589) < clean-208 (0.7750) < nnU-Net (0.8187): the prediction failed.** Adding 222 masked
  single-side patients did not exceed the 208 clean both-annotated cases, let alone beat 0.8187. §21's
  "should beat 0.8187" does **not** hold on this test set.
- **Caveats:** n=43 single fold/seed, both deltas within/near the ≈0.05 paired-noise floor; the dirty control
  **diverged** (best ckpt @ ep45) so masked-vs-dirty is confounded by training stability; and the test set is
  all-both-parotid — structurally unable to reward the masked loss's main expected benefit (single-side
  cases). Interpret as suggestive: masking helps vs the naive control and trains more stably, but is not the
  0.82-beating fix §21 predicted. Full record + hypotheses: `ablation_study/E6_masked_loss/RESULT.md`.

---

### 20.F  Single-side evaluation + per-side specialist experts (E7) — ✅ DONE (2026-07-18)
_(The E7 agent appends here — task spec in `ablation_study/E7_singleside_eval_and_experts_tasks.md`. Two parts:
**Part 1** (free, no training) — evaluate all existing models on the ~58 held-out **single-side** test
patients that the study never measured (the "easy subset" gap, SYNTHESIS §9.8), scoring the annotated side +
a contralateral-prediction-rate mechanism figure. **Part 2** (optional GPU, ~$3–6) — **per-side specialist
models** (a left-parotid expert on ~320 patients, a right-parotid expert on ~318), the nnU-Net-native
"decompose" treatment of the annotation gap that completes the four-way set (discard/penalise/mask/decompose);
the agent must critique/improve the design — incl. a shared-backbone / flip-symmetry / region-based single
model alternative — before running. NOT a "mixture of experts" (no gating). Append: single-side metrics,
contralateral rates, experts' both-parotid + single-side results, cost, verdict, caveats.)_

- ✅ Part 1 — single-side eval (existing models) — **DONE (2026-07-18).** Verified from raw data: TEST split =
  44 both / 30 only_R / 28 only_L / 63 none ⇒ **58 single-side**. Leakage clean (disjoint 667-patient train/val
  pool). **L/R naming validated 58/58 vs geometry.** eval harness reproduces nnU-Net **0.8187** exactly.
- ✅ Part 2 — per-side experts — **DONE (2026-07-18).** Chose Option A: two single-class nnU-Net experts
  (left **320**, right **318**), trained 2× A40 parallel, 250-epoch noMirror, patch [48,224,192]/b2 — config
  identical to clean-208. Combined via hemifield tie-break. Rejected flip-symmetry (mirroring disabled, §15) and
  native ignore-label single model (= E6/masked-430 family, already 0.7589 < 0.8187).

**E7 RESULTS (locked test set):**

| set | metric | nnU-Net clean-208 | E7 experts | E4 | dirty-430 | masked-430 |
|---|---|---|---|---|---|---|
| both-parotid 43 | Dice | **0.8187** | 0.8099 | 0.775 | — | 0.7589 |
| single-side 58 | annot Dice+CC | 0.8523 | **0.8552** | 0.8011 | 0.7436 | 0.8010 |
| single-side 58 | contra-rate | 96.6% | 98.3% | 100% | **87.9%** | 94.8% |

- **Both-parotid:** experts (0.8099) competitive but **do not beat 0.8187** — completes the four-way gap set
  (discard 0.8187 / penalise dirty / mask 0.7589 / **decompose 0.8099**); no strategy beats *discarding* the
  single-side data on the easy set.
- **Single-side (the hole SYNTHESIS §9.8 flagged, now filled):** experts are **top (0.8552)**, narrowly over
  clean-208; the at-risk annotated gland scores **~0.85 — higher per-gland than the both-side average**, so
  single-side is NOT the feared catastrophe. Combining experts is **lossless** (combined per-side = standalone).
- **Annotation-gap signal:** dirty-430 suppresses the healthy contralateral gland most (87.9% vs 95–100%),
  masking recovers it (94.8%) — §16 confirmed **directionally but modest** (all models predict it ≥88%).
- **Cost:** ~$8 of the $9.58 balance (training ≈$3.4; rest = a troubled upload — see POD_UPLOAD_PLAYBOOK note on
  splitting only-missing files across streams). Detail: `E7_singleside/RESULT.md`, `E7_experts/RESULT.md`.

### 20.G  Prediction gallery (E8) — ✅ DONE (2026-07-18; pure rendering, $0, no GPU)
Qualitative test-set prediction visualisations for every important model, rendered locally from the existing
prediction files (no training/inference). Output: **`ablation_study/prediction_gallery/`** (open `index.html`).
- **Per-case** axial montages (CT soft-tissue window + pred R=red/L=blue fill+contour + GT dashed, per-case
  Dice in title) **and** 2-view 3D marching-cubes surface renders, for **15 models × 43 both-parotid cases**
  (645 montages + 645 3D): nnU-Net 1-fold/5-fold, E1 2d, E2 dirty-430, E2b gapped-208, E4 +CC, E6 masked-430
  +CC, E6 dirty-430 +CC, E7 experts (combined + left-only + right-only), and the P0 baselines (U-Net,
  Attention, TransUNet, Swin).
- **Same-case-across-models comparison grids** (the headline output) for 6 representative cases (best PAR0228,
  median PAR0222, worst PAR0223, max-disagreement PAR0214, quartiles PAR0247/PAR0237): one axial slice, every
  model side-by-side, GT dashed in each — plus a separate P0-baselines grid per case.
- **Best-of showcase** (`prediction_gallery/showcase/showcase.html`): each model's top-8 cases by Dice +
  a hall-of-fame of the 15 highest-Dice full both-parotid predictions in the study (led by nnU-Net
  PAR0228 at 0.920). Montage slice panels are spread over the **GT z-extent** so every panel is centred
  on a real contour (an earlier union-based spread put end panels on one-sided gland-tip slices that
  looked like misses despite high Dice — verified a slice-choice artifact, not misalignment: pred/GT
  centroids agree <1 voxel). Render style adopted from the website renderer
  (`Parotid-Project/Website/_parotid-render-tools/render_cases.py`): black bg, head-cropped panels,
  bilinear CT, cream GT dashed, gaussian-smoothed shaded `plot_trisurf` 3D.
- Verified per-case Dice reproduces §20.0 aggregates **exactly** for all 11 `.nii.gz` models. **Geometry note:**
  the P0 `.npz` baselines were expected in a different geometry, but their GT matches `Dataset002/labelsTs`
  **exactly (Dice 1.000)** — identical 512×512×114 grid — so they are overlaid on the same CT and are directly
  comparable (still shown in a separate section). Colour/label convention R=red #ff3b30 / L=blue #0a84ff / GT
  dashed throughout; dpi 200. New files live only under `prediction_gallery/`; see its `README.md`. Re-runnable
  scripts: `galcommon.py`, `compute_metrics.py`, `render_all.py`, `make_comparison.py`, `render_p0.py`,
  `build_index.py`.

---

*End of Section 22 (plus the §20.E E6 result and §20.F E7 stub). This master document is the complete record
of the project — dataset, code, Phase-1/Phase-2 results, the L/R bug, the full ablation, the tested (and
refuted) masked-loss prediction, the verified corrections, and the planned single-side eval + per-side
experts. Nothing about the project should require reading the folder from scratch again.*
