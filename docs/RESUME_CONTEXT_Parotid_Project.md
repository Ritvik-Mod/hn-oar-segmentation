# Resume Context — Parotid Gland Auto-Segmentation Project

> Complete factual brief for updating a resume / CV. Covers the whole project end
> to end, quantified results, technical skills demonstrated, ready-to-use bullet
> points, and honest scope notes. Owner: 2nd-year B.Tech CSE student, BIT Mesra.
> Solo project (~1 month of focused work), built on a dataset the owner assembled
> and processed himself under a radiation oncologist's clinical supervision.

---

> **⚠️ UPDATED 2026-07-17 after the controlled ablation study.** The old headline "0.62 → 0.82" is
> **retired** — the 0.62 was a *validation* number and the 0.82 a *test* number, so ~62% of that "jump"
> was a measurement artifact. The honest figure is **0.743 → 0.819 (+0.075) on the same held-out test
> set.** The ablation is now the project's strongest asset (it caught this). Canonical numbers live in
> `PROJECT_NARRATIVE.md`; full evidence in `MASTER_PROJECT_REFERENCE.md` §20. Use the numbers below.

## 1. ONE-LINE SUMMARY
Built an end-to-end deep-learning system that automatically segments the parotid glands (a critical
organ-at-risk) on head-and-neck CT for radiotherapy planning — from reverse-engineering the raw hospital
contour format to a 3D nnU-Net reaching **0.82 Dice / 5.2 mm HD95** on a held-out test set — **and then ran
an 8-experiment controlled ablation that dissected the result**, catching a validation-vs-test measurement
artifact that had inflated the reported improvement and showing that **label quality, not architecture,
drives performance** on partial-label clinical data.

## 2. PROBLEM / MOTIVATION (domain context)
- In head-and-neck cancer radiotherapy, the tumour needs a high radiation dose
  (≥60 Gy) but the parotid (salivary) glands tolerate far less (~20–26 Gy);
  over-dosing them causes xerostomia (permanent dry mouth). Oncologists must
  manually outline ("contour") these glands on every CT slice — ~30 minutes per
  patient across ~10 organs, at a hospital treating ~800–1,000 such patients/year.
- The partner hospital had no auto-contouring software. Goal: a model that produces
  clinically usable parotid contours to accelerate this workflow (decision-support,
  not autonomous diagnosis).

## 3. DATA ENGINEERING (a major part of the work)
- **Reverse-engineered a proprietary contour format.** The raw exports from the
  hospital's Elekta Monaco treatment-planning system store contours in an
  undocumented `.WC` plaintext format (no public spec). Built a parser from scratch
  that reconstructs each contour in pixel space using affine transforms derived from
  DICOM metadata (ImagePositionPatient, PixelSpacing). Also built a second parser
  for standard DICOM-RTSTRUCT exports.
- **Assembled and standardized a 914-patient dataset** (~127,000 annotated CT
  slices). Wrote a label-standardization pass mapping dozens of inconsistent
  structure-name variants to canonical names and dropping hardware/artifact labels.
- Stored as per-slice compressed arrays (512×512 16-bit CT image + binary masks per
  structure). Established a locked, patient-level train/val/test split (583/84/165,
  fixed seed) so evaluation stays honest.
- Identified and handled real data-quality issues: mislabeled/corrupt masks, and
  inconsistent/partial annotations (clinicians contour only clinically relevant
  organs — e.g. only the at-risk side).

## 4. FIRST MODELING PHASE — custom architectures (baseline)
- Implemented four segmentation architectures **from scratch in PyTorch**: U-Net
  (~31M params), Attention U-Net (~31M), TransUNet (CNN-transformer hybrid, ~102M),
  and Swin-UNet (pure transformer, ~27M). Custom combined Dice + BCE loss, weighted
  sampling for class imbalance, mixed-precision training on a university HPC cluster
  (2× NVIDIA L40, PBS scheduler).
- **Key finding:** all four clustered together regardless of parameter count (31M to 102M) — ~0.62 3D Dice
  on the *validation* set, or **0.72–0.74 when later re-scored on the held-out test set** (the ablation
  showed the validation numbers were deflated by the annotation gap). Either way the models are within
  ~0.03 Dice of each other, diagnosing the bottleneck as the *data/preprocessing pipeline* (raw 2D slices,
  inconsistent voxel spacing, no resampling, partial labels), **not** model capacity — a mature insight
  that the later ablation confirmed quantitatively.

## 5. SECOND MODELING PHASE — nnU-Net + full 3D pipeline (the deliverable)
Rebuilt the approach around **nnU-Net v2** (the self-configuring SOTA medical-
segmentation framework), which required a complete new 3D pipeline:
- **3D volume reconstruction:** turned the per-slice arrays into proper 3D NIfTI
  volumes, recovering correct anisotropic voxel spacing (in-plane 0.977 mm, 3.0 mm
  slices) from physical slice positions; robust to missing/duplicate slices.
- **nnU-Net dataset conversion** with the locked test set held out; automated QC to
  drop corrupt masks; consistent left/right label handling.
- **3D evaluation framework** computing Dice, clinical Tversky, HD95, and Surface-
  Dice with true anisotropic spacing (validated against known-correct cases).
- **Masked partial-label loss** (for a future multi-organ extension); reviewed and
  fixed a correctness bug in a drafted version and unit-tested it.

## 6. COMPUTE / MLOps ENGINEERING
- Hit GPU contention on the shared university cluster (both L40s saturated by other
  users → out-of-memory) and **migrated training to rented cloud GPUs (RunPod)**.
- Trained nnU-Net 3d_fullres on a rented **H100**; then trained a **5-fold ensemble
  by running four folds in parallel on a 4× A40 machine** (custom trainer, explicit
  per-GPU pinning, tmux, live logging), finishing in ~2 hours for ~$6.
- Handled real logistics: institutional firewall blocking cloud access to the HPC
  (relayed 7 GB of data via the local machine), persistent network volumes, resumable
  transfers, and cost control.

## 7. THE KEY DEBUGGING RESULT (strongest interview story)
- The first nnU-Net evaluation on the held-out test set came back **worse than the
  baselines**: Dice 0.50, HD95 65 mm — a red flag given training looked healthy.
- **Diagnosed it with a controlled ablation:** a left/right "swap test" showed 11 of
  44 cases scored near-zero normally but recovered to 0.6–0.85 when the predicted
  left/right labels were swapped; the left/right-agnostic Dice was already 0.82.
- **Ruled out the obvious cause:** re-labeling the ground truth by geometry corrected
  only 1 of 251 cases → the annotations were consistent; the model was the problem.
- **Found the root cause:** nnU-Net's default mirror (left-right flip) data
  augmentation and test-time mirroring make left/right-distinct structures
  interchangeable — a known but easy-to-miss failure mode. Confirmed by isolating it:
  disabling test-time mirroring alone only recovered 0.50→0.56 (HD95 still 64 mm), so
  the confusion was baked into the weights.
- **Fixed it** with a custom nnU-Net trainer that disables mirroring. The test score
  jumped to 0.82 Dice / 5.2 mm HD95. This whole diagnose→isolate→fix chain is the
  most compelling part of the project.

## 8. FINAL RESULTS (all on the SAME locked held-out test set, 43 both-parotid patients)
| Model | 3D Dice | HD95 (mm) | Surface-Dice (3 mm) |
|---|---|---|---|
| Best from-scratch baseline (Attention U-Net, 2D) — **on test** | 0.7434 | 3.57 | 0.8827 |
| nnU-Net single fold (no-mirror) | 0.8187 | 5.25 | 0.8965 |
| **nnU-Net 5-fold ensemble (final)** | **0.8202** | **5.24** | **0.8990** |
| Per-side (ensemble) | R 0.833 / L 0.808 | R 5.14 / L 5.34 | — |

- **Honest headline: 0.743 → 0.819 Dice (+0.075) on the same held-out test set.** (The retired "0.62 → 0.82"
  compared a *validation* baseline to a *test* result; on the same test set the baseline is 0.743, so ~62%
  of that apparent jump was a measurement artifact — a correction the ablation surfaced.)
- The separate L/R mirror-bug fix (§7) is a real 0.50 → 0.82 recovery on test — keep that story; it is not
  the same comparison as the baseline-vs-nnU-Net gain above.
- The 5-fold ensemble is **not** significantly better than a single fold (p≈0.35) — ship the single fold.

## 8b. THE ABLATION (the project's strongest, most original result)
An 8-experiment controlled study (one variable each, one locked test set) decomposed the real +0.075:
- **Label quality is the dominant driver** (+0.046 net, +0.129 when isolated) — ahead of preprocessing
  (+0.022), dimensionality/3D (+0.007) and ensembling (+0.0015, n.s.).
- **Architecture is near-irrelevant:** six 2D models span just **0.028** Dice; ImageNet pretraining adds
  +0.006 (transformers still lose to a from-scratch CNN); a hand-built 3D U-Net + trivial post-processing
  matches nnU-Net's boundary accuracy.
- **The star finding — the annotation gap:** clinicians deliberately leave the healthy-side parotid
  un-contoured (**51.6%** of parotid patients are single-side); this costs ~0.13 Dice in *training* AND
  deflates *validation* scores by ~0.13. Clinically validated by the advisor; genuinely novel vs the
  literature.
- Take-away line: **"the data is the product, not the model."**
- **Extended the study (E6/E7) with a full four-way treatment of the annotation gap** — discard / penalise /
  mask / decompose. Built and tested a **masked partial-label loss** (beats the naive control + stabilises
  training, but doesn't beat the clean baseline) and **per-side specialist models** (competitive, 0.810 vs
  0.819). **No strategy for using single-side data beat simply using the clean both-parotid set** — data
  quality > quantity. A **held-out single-side evaluation** (58 patients the study never measured) showed the
  clinically-relevant at-risk gland scores **~0.85** — so real-world one-sided performance is strong. Making a
  number-backed prediction, testing it, and reporting the miss is the rigorous outcome (good interview story).

## 9. DELIVERABLES PRODUCED
- A structured, reproducible repository (`Parotid_Segmentation_Complete/`): parsers,
  preprocessing, model code/checkpoints, results, inference demo, and documentation.
- **Interactive result galleries** (per-case CT overlays: predicted vs. oncologist
  contours + 3D gland renders + per-case metrics), and a single self-contained HTML
  report for sharing.
- A **Gradio inference demo** for viewing predictions on new scans.
- Full written documentation including a diagnostic case file of the L/R issue.

## 10. TECHNICAL SKILLS DEMONSTRATED (keywords for the resume)
- **ML/DL:** PyTorch, 3D CNNs, semantic segmentation, U-Net / Attention U-Net /
  TransUNet / Swin-UNet, nnU-Net v2, transfer of framework internals (custom
  trainers), data augmentation, cross-validation, model ensembling, loss design
  (Dice/BCE/Tversky, masked partial-label loss), class imbalance handling.
- **Medical imaging:** CT/DICOM, RTSTRUCT, NIfTI, HU windowing, anisotropic
  resampling, nibabel/SimpleITK, segmentation metrics (Dice, HD95, Surface-Dice),
  radiotherapy organ-at-risk contouring domain knowledge.
- **Data engineering:** reverse-engineering a proprietary binary/text format, large-
  scale ETL (~127k slices / 914 patients), label standardization, automated QC.
- **MLOps / infra:** Linux, HPC/PBS job scheduling, cloud GPU provisioning (RunPod),
  multi-GPU parallel training, tmux, SSH/scp/rsync, cost/experiment management.
- **Engineering judgment:** systematic debugging via ablation, hypothesis testing,
  honest held-out evaluation, reproducibility.

## 11. READY-TO-USE RESUME BULLETS (pick/trim as needed)
- Built an end-to-end 3D deep-learning pipeline for parotid-gland segmentation on head-and-neck CT reaching
  **0.82 Dice / 5.2 mm HD95** on a held-out test set (a **+0.075** improvement over strong from-scratch
  2D baselines, scored on the same locked test set).
- **Ran an 8-experiment controlled ablation that caught a validation-vs-test measurement artifact inflating
  the reported gain by ~62%, then attributed the true improvement across six confounded axes** — showing
  label quality, not architecture, dominates (six models within 0.028 Dice; ImageNet pretraining +0.006;
  ensembling not significant).
- **Quantified a clinically-validated "annotation gap"** (clinicians skip the healthy-side parotid in 51.6%
  of cases) costing ~0.13 Dice on both the training and evaluation sides — the study's most novel finding.
- Reverse-engineered an undocumented proprietary radiotherapy contour format and
  built parsers to assemble a **914-patient, ~127k-slice** annotated CT dataset,
  including label standardization and automated quality control.
- Diagnosed a subtle failure where left/right glands were being confused (test Dice
  stuck at 0.50); isolated the cause to mirror data-augmentation via a controlled
  swap-test ablation and fixed it with a custom nnU-Net trainer — recovering full
  performance.
- Implemented and benchmarked four segmentation architectures from scratch in
  PyTorch (U-Net, Attention U-Net, TransUNet, Swin-UNet) and showed the accuracy
  ceiling was data-pipeline-bound, not architecture-bound.
- Ran distributed training on university HPC and rented cloud GPUs (H100, 4× A40
  parallel folds), managing scheduling, multi-GPU execution, data transfer, and cost.

## 12. HONEST SCOPE / CAVEATS (so nothing is overstated in interviews)
- Single-institution dataset; parotids only (the pipeline is built to extend to more
  organs, but only parotids are trained/evaluated so far).
- Final test evaluation is on the 43 test patients with both parotids annotated
  (corrupt cases QC-excluded); it is a research/portfolio result, not a clinically
  deployed or regulatory-cleared tool.
- Decision-support framing (assists contouring); not autonomous diagnosis.
- nnU-Net is an existing framework — the contribution is the data pipeline, the
  debugging, the custom trainer, and the end-to-end delivery, not the base
  architecture.
- **Small-n honesty:** the test set is n=43 (both-parotid, QC-clean — the "easy" subset), single fold /
  single seed, so cross-run deltas below ~0.02–0.05 Dice are within noise (the 3D, pretraining and
  ensembling effects are "no measurable effect", not zero). 0.82 is an optimistic operating point; nothing
  here measures single-side or messy-case performance. Findings are a single-institution case study — the
  transferable lessons are hypotheses for other cohorts, not proven laws.

## 13. SUGGESTED PROJECT TITLE / TAGLINE OPTIONS
- "Automated Organ-at-Risk Segmentation for Head-and-Neck Radiotherapy (3D nnU-Net)"
- "Deep-Learning Parotid Segmentation on CT — data pipeline to 5-fold ensemble"
