# Deep Learning Pipeline for Automated Head & Neck Organ-at-Risk Segmentation in Radiotherapy

> **Active research project.** Code, model weights, and quantitative results will be released upon paper acceptance. This repository currently contains documentation only.

---

## Overview

Intensity Modulated Radiation Therapy (IMRT) for head and neck (H&N) cancer achieves submillimetre dose precision, but that precision is only as good as the organ contours it targets. Manual delineation of organs-at-risk (OARs) by a clinical oncologist takes **30–90 minutes per patient** and introduces significant inter-observer variability. Inaccurate parotid gland contouring, in particular, directly causes **xerostomia** (chronic dry mouth), a debilitating, often permanent side effect that significantly reduces quality of life for H&N cancer survivors.

This project develops and benchmarks a fully automated deep learning pipeline for parotid gland segmentation from CT scans, trained on an **Indian-demographic clinical dataset**, a population that is critically underrepresented in existing H&N OAR segmentation literature.

---

## Dataset

| Property | Details |
|---|---|
| **Confirmed H&N radiotherapy patients** | 844 |
| **Usable patient folders (after QC)** | 914 |
| **Institution** | Single Indian tertiary cancer centre |
| **Treatment Planning System** | Elekta Monaco TPS |
| **Raw data formats** | 797 patients in proprietary `.WC` format + 117 patients in DICOM-RT |
| **Annotated CT slices** | 126,879 |
| **HDF5 database size** | 38 GB |
| **Demographic** | Indian population; underrepresented in published H&N OAR benchmarks |
| **Ethics** | Clinical data used under institutional ethics review |

This dataset contributes to **demographic diversity in medical AI**: the vast majority of published H&N segmentation models are trained and validated exclusively on Western demographic cohorts, raising concerns about cross-population generalisation.

---

## Data Engineering Pipeline

The engineering challenge of this project was substantial before any model training began.

### Format 1: Proprietary `.WC` Reverse Engineering

The majority of the dataset (797 patients) existed exclusively in Elekta Monaco's proprietary `.WC` file format, for which **no public documentation, parser, or specification exists anywhere**. The format was reverse-engineered entirely from first principles through binary analysis, pattern recognition across multiple patient files, and iterative hypothesis testing.

Key outcomes:
- Developed a custom binary parser that reliably extracts physician-annotated organ contours across all 797 patients
- Derived a **custom affine coordinate transformation** mapping physical millimetre coordinates (from the treatment planning space) to pixel-space indices in the corresponding CT volume
- Validated extraction correctness against DICOM-RT ground truth on overlapping cases

### Format 2: DICOM-RT Pipeline

A separate extraction pipeline processes the 117 DICOM-RT patients, handling:
- RT Structure Set (RTSTRUCT) parsing via `pydicom`
- Contour-to-mask rasterisation across all referenced CT slices
- Coordinate system alignment between RTSTRUCT and CT image frames of reference

### Data Consolidation & Quality Control

- Both pipelines feed into a single unified **38 GB HDF5 database** (via `h5py`) for fast random-access training
- **Statistical anomaly analysis** on mask pixel distributions identified and excluded **12 corrupt patients** whose annotation data could not be reliably reconstructed
- A **label standardisation pass** was run across all 126,879 slices with zero labelling errors
- **Patient-wise train/val/test split** locked with `seed=42` before any model training: **583 / 84 / 165 patients**, preventing any form of data leakage

### Performance Optimisation

| Task | Before | After |
|---|---|---|
| Full dataset extraction | 9+ hours (sequential) | < 2 hours |
| Method | Single-process | `ProcessPoolExecutor` (8-core parallel) |

### Dataloader Design

- **HU windowing**: −150 to +250 HU applied at load time (not baked into stored files, preserving flexibility)
- **Normalisation**: Min-max to [0, 1] after windowing
- **Class imbalance**: `WeightedRandomSampler` with a **20:1 parotid-to-background ratio**
- **Augmentation**: Horizontal flip with **anatomically-correct left/right mask channel swap**; standard random flip would laterally invert the parotid label assignment, which is anatomically incorrect and would corrupt training

---

## Architecture Benchmark

Five architectures are evaluated under **strictly controlled, identical conditions**: same loss function, same optimiser, same hyperparameters, same data split. No pretrained weights are used across any architecture. This is a controlled benchmark isolating architectural differences as the sole variable.

| Architecture | Type | Key Characteristic |
|---|---|---|
| **U-Net** | CNN | Established encoder-decoder baseline |
| **Attention U-Net** | CNN + Attention | Soft attention gates on skip connections |
| **TransUNet** | Hybrid CNN–Transformer | ViT encoder on CNN feature maps |
| **Swin-UNet** | Pure Transformer | Shifted-window self-attention throughout |
| **nnU-Net** | Self-configuring | Automatic hyperparameter adaptation |

**Training protocol (identical across all architectures):**
- Loss: `0.5 × Dice + 0.5 × BCE` (with sigmoid activation)
- Optimiser: Adam, lr = 1e-4
- Early stopping: patience = 10 epochs
- Data split: same locked 583/84/165 patient-wise split

> Quantitative results are withheld pending publication.

---

## Evaluation Framework

Each trained model is evaluated using a clinical evaluation suite computing the following metrics on the held-out test set (165 patients, 3D volumetric):

| Metric | Description |
|---|---|
| **3D Volumetric Dice** | Overlap between predicted and ground-truth volume |
| **HD95 (mm)** | 95th-percentile Hausdorff Distance; captures boundary outliers |
| **Surface Dice (3 mm tolerance)** | Clinically-relevant surface agreement within a 3 mm margin |
| **Clinical Tversky (α=0.3, β=0.7)** | Custom asymmetric metric penalising under-segmentation more heavily than over-segmentation |

The asymmetric Tversky metric reflects a clinical priority: in radiotherapy planning, **under-segmenting a parotid gland is more dangerous than over-segmenting it**, since under-contouring means the organ receives radiation that the plan assumed it would not.

---

## Tech Stack

```
Languages:     Python 3.10+
DL Framework:  PyTorch 2.x
Data:          HDF5 (h5py), NumPy, SciPy
Medical I/O:   pydicom, custom .WC parser
CV / Imaging:  OpenCV, skimage
Parallelism:   concurrent.futures.ProcessPoolExecutor
```

See [`requirements.txt`](requirements.txt) for the full dependency list.

---

## Project Status

```
[✅] Data extraction: WC pipeline            Complete
[✅] Data extraction: DICOM-RT pipeline      Complete
[✅] HDF5 consolidation & QC                  Complete
[✅] Dataloader with augmentation             Complete
[✅] Clinical evaluation framework            Complete
[✅] U-Net training & evaluation              Complete
[✅] Attention U-Net training & evaluation    Complete
[✅] TransUNet training & evaluation          Complete
[✅] Swin-UNet training & evaluation          Complete
[🔄] nnU-Net                                  In progress
[⏳] Ablation studies                         Pending
[⏳] Final benchmark analysis                 Pending
[⏳] Paper submission (ISBI / MIDL)           Pending
```

**Code and results will be made publicly available upon paper acceptance.**

---

## Ethics & Data Use

This project uses retrospective clinical CT and radiotherapy planning data from an Indian tertiary cancer centre, collected under institutional ethics review. No patient-identifiable information is present in any processed file. The dataset is not publicly shareable.

Contributing to demographic diversity in medical AI: the model is developed and validated on an Indian-demographic cohort, addressing a known gap in H&N OAR segmentation literature where virtually all published benchmarks use Western (US/European) patient populations.

---

## Citation

Paper in preparation. Citation details will be added upon publication.

```bibtex
@article{mod2026hnOAR,
  author    = {Mod, Ritvik and others},
  title     = {Deep Learning Benchmark for Automated Head and Neck
               Organ-at-Risk Segmentation on an Indian Demographic CT Dataset},
  journal   = {[Venue TBD]},
  year      = {2026},
  note      = {Under review}
}
```

---

## Author

**Ritvik Mod**  
B.Tech Computer Science and Engineering, BIT Mesra (2024–2028)  
[ritvikmod@gmail.com](mailto:ritvikmod@gmail.com)

**Clinical Advisor:** Senior Radiation Oncologist *(name withheld pending paper submission)*

---

*This repository is intentionally sparse. The engineering and research work described above is real and ongoing. Code will be open-sourced upon paper acceptance.*
