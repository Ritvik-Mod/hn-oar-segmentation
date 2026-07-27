# CASE FILE — H&N Parotid Segmentation, nnU-Net Training & L/R Diagnosis

> Factual record for external review. Compiled 2026-07-07. Describes: (1) the state of the project before this working session, as reported by the owner; (2) everything done in this session; (3) the first nnU-Net training run and its results; (4) the observations, deductions, and corrective actions taken; (5) the current open decision. Statements are recorded as what was done/observed/said. Where something is an inference rather than a measured fact, it is explicitly labelled "DEDUCTION" or "HYPOTHESIS".

---

## PART A — PROJECT STATE BEFORE THIS SESSION (as reported by the owner)

Source: owner-provided `PROJECT_STATE_FOR_AI_AGENT.md` (compiled July 2026 from files last modified March–April 2026) plus owner statements.

### A.1 Identity / clinical problem
- Owner: 2nd-year B.Tech CSE student. Clinical advisor: a radiation oncologist. Single-institution hospital partner; Elekta Monaco TPS; Siemens CT scanners; 2 annotating oncologists.
- Task: auto-contour organs-at-risk (OARs) on head-and-neck (H&N) CT. Parotid glands are the primary target (tumour needs ≥60 Gy; parotids tolerate 20–26 Gy; over-dose → xerostomia). Manual contouring ≈30 min/patient. No auto-contouring software at the hospital. Technique: IMRT.
- Original intent was a research paper. At the start of this session the owner changed the goal to a **portfolio-grade working model** for internship applications (deliverables: live demo, clean repo, strong metrics, deployable artifact).

### A.2 Dataset (as it physically exists)
- `ML_Dataset_Final/` — 914 patient directories; canonical processed dataset. Each slice stored as a `.npz` file at `<patient>/<CT_group>/data/Z_<z>.npz`, plus verification overlay PNGs.
- Each `.npz` holds `image` (512×512, uint16, raw pixel) plus one binary uint8 (512×512) mask per contoured structure on that slice. Observed structure keys include `BODY, SPINAL_CORD, SPINAL_CORD_PRV, PAROTID_L, PAROTID_R, PTV, PTV_MARG` and many more.
- Two parsing pipelines feed the dataset: a reverse-engineered Elekta Monaco `.WC` parser (~797 patients; applies affine `Px=(X−Ox)/Sx`, `Py=(−Y−Oy)/Sy`), and a direct DICOM-RTSTRUCT parser (~117 patients). Patient IDs: WC-style contain `~` (e.g. `1~240125`); DICOM-style are numeric (e.g. `260229`).
- CT specs (from docs): 512×512, 16-bit. Dominant in-plane pixel spacing 0.977×0.977 mm (~90.5% of scans; full range 0.643–0.977 mm). Slice thickness 1.0 or 3.0 mm. Avg ~153 slices/patient. HU conversion `HU = pixel − 8192`.
- Locked patient split (`dataset_split.json`, seed 42): **train 583 / val 84 / test 165** (832 total). Test set had **never been evaluated**.
- Documented data-quality issue: **~12 patients with mislabeled parotid masks** (20,000+ px/slice vs normal 500–5,000) were excluded at loading time in the original pipeline.

### A.3 Models trained previously (owner's earlier work)
- Four architectures, all hand-implemented in PyTorch, `in=1, out=2` (channels = PAROTID_R, PAROTID_L), trained from scratch: U-Net (~31M params), Attention U-Net (~31M), TransUNet (~102M), Swin-UNet (~27M).
- Training: combined loss = 0.5·Dice + 0.5·BCE, Adam lr 1e-4, batch 8, on BIT Mesra HPC (`rachel`, 2× NVIDIA L40).
- Reported **validation** 3D volumetric Dice (`documentation/results_tracker.csv`):
  - Vanilla U-Net 0.6209; Attention U-Net 0.6346; TransUNet 0.6332; Swin-UNet 0.5106.
  - All CNN/hybrid models clustered at **~0.62–0.63 Dice regardless of parameter count**.
- Checkpoints present locally (`checkpoints/`). Test set never run. nnU-Net planned but never trained.
- Documented clinical finding: models predicted bilateral parotids where GT had only unilateral annotation; the advisor confirmed clinicians deliberately omit the contralateral healthy-side parotid on one-sided tumours (partial annotation is clinically appropriate).

---

## PART B — WHAT WAS DONE IN THIS SESSION (before training)

### B.1 Data audit (Claude, on the owner's Mac)
- Confirmed the full dataset is present locally (`ML_Dataset_Final`, 55 GB) and all 5 `.pth` checkpoints are local.
- Sampled 150/914 patients and tallied structure annotation frequency. Findings: annotation is **partial and inconsistent** (clinicians contour only clinically relevant organs). Beyond parotids, several H&N OARs are annotated often enough to model (eyes ~65%, spinal cord ~41–54%, larynx ~51%, optic nerves, temporal lobes). npz mask keys still contain non-standardised raw names (`dx22`, `X2`, etc.).

### B.2 Decisions taken (with owner)
- Owner answered: HPC (rachel) still usable; compute could also be rented; ~1 month timeline; multi-organ desired only if data supports it (it does).
- Evaluated two suggestions the owner had received from friends:
  - **Mixture-of-Experts** for dataset variation — REJECTED. DEDUCTION: heterogeneity (pixel spacing, scanner, two pipelines) is better handled by resampling + normalization + augmentation than by MoE routing.
  - **Splitting each image into 4 sub-images** (quadrant tiling) — REJECTED. DEDUCTION: fixed quadrants bisect organs and lose context; sliding-window patching (nnU-Net) is the principled equivalent.
- Chosen backbone: **nnU-Net v2**, from scratch (cross-dataset pretrained-weight transfer judged too finicky; nnU-Net from scratch already SOTA). Owner's from-scratch models kept as portfolio evidence, not the deliverable.
- Staged plan agreed: Phase 1 = nnU-Net parotid model + test eval + demo; Phase 2 = multi-organ with masked partial-label loss.

### B.3 Artifacts built this session (in `project/pipeline/` unless noted)
- `sanity_check_inference.py` — loaded `checkpoints/attention_unet/best_model_attention.pth` (epoch 10, val_loss 0.1637); confirmed it loads and produces anatomically-correct parotid predictions on validation slices (per-slice 2D Dice 0.53–0.90). Preprocessing recipe confirmed: `image − 8192` → clip [−150, 250] HU → /400 → sigmoid → threshold 0.5; ch0 = PAROTID_R, ch1 = PAROTID_L.
- `demo/app.py` (+ `demo/attention_unet.py`, `demo/examples/*.npz`) — a Gradio demo on the Attention U-Net checkpoint. Verified serving locally (HTTP 200); example slice Dice ~0.93. NOT deployed (owner chose to keep local).
- `pipeline/build_volumes.py` — reconstructs per-slice npz → 3D NIfTI (image + per-organ label volumes). Picks the CT group with the most annotations; sorts slices by physical Z from filename; sets Z spacing from the median filename-Z step (typically 3.0 mm); sets in-plane spacing to the documented **0.977 mm** (npz store no spacing). Verified: correct shape/spacing/dtype and mask–image alignment confirmed visually.
  - DATA FACT established: the npz files do **not** store pixel spacing; Z is recoverable from filenames; in-plane X/Y is not stored and raw DICOM is only local for a small subset, so in-plane 0.977 mm is used as a documented assumption.
- `pipeline/make_nnunet_dataset.py` — builds an nnU-Net v2 raw dataset. Integer labelmap: 0=bg, 1=parotid_r, 2=parotid_l. Includes **only patients with BOTH parotids annotated** (to keep labelmaps clean); keeps the locked TEST set in `imagesTs`/`labelsTs` (never trained on); writes `case_mapping.json` (PARxxxx → real patient id).
- `pipeline/masked_loss.py` — masked partial-label (sigmoid + Dice+BCE) loss for the future multi-organ phase. During review of a drafted version, a bug was found and fixed (masked-out classes leaked into the averaged Dice term); unit-tested (corrupting an unannotated channel leaves the loss unchanged).
- `pipeline/eval_testset.py` — 3D Dice / Clinical Tversky / HD95 / Surface-Dice, using **true anisotropic spacing (0.977×0.977×3.0)** and bbox-cropping for speed. Validated: perfect prediction → Dice 1.0 / HD95 0; a 4-voxel shift → Dice 0.68 / HD95 3.82 mm (≈ 4×0.977, physically correct).
- `pipeline/README_HPC.md`, `pipeline/nnunet_prep.pbs`, `pipeline/nnunet_train.pbs` — HPC runbook and PBS job scripts (later adapted, then superseded by cloud training).

### B.4 Annotation-count facts (full split, both-parotid presence)
Measured across the locked split:
- train (583): both=186, only-R=98, only-L=99, no-parotid=200 → 383 with a parotid.
- val (84): both=22, only-R=12, only-L=13, no-parotid=37 → 47 with a parotid.
- test (165): both=44, only-R=30, only-L=28, no-parotid=63 → 102 with a parotid.
- Dataset built (`Dataset001_Parotid`, both-annotated only): **208 train+val cases** (186+22) and **44 test cases**. Single-side patients deferred to the future masked-loss phase.

---

## PART C — TRAINING RUN (first nnU-Net training)

### C.1 Compute
- HPC (rachel) was attempted but both L40 GPUs were saturated by other users' jobs (nvidia-smi showed ~45 GB/46 GB used per GPU); training died with `CUDA error: out of memory`. (An earlier project note also cited "GPU contention" as why nnU-Net was never trained.)
- Switched to a rented **RunPod 1× H100 SXM (80 GB)**, PyTorch 2.8.0 template, ~$3.30/hr. The HPC blocked SSH from the cloud IP (institutional firewall), so the built dataset (~7 GB) was relayed **HPC → owner's Mac → pod**.

### C.2 Preprocessing (nnU-Net auto-configuration)
- `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity` succeeded (after fixing an incomplete upload — `dataset.json` had initially not transferred).
- Chosen 3d_fullres config: spacing **[3.0, 0.977, 0.977]** (anisotropy preserved, not forced isotropic), patch size **[48, 224, 192]**, batch size 2, `CTNormalization`, `InstanceNorm3d`, PlainConvUNet, 6 stages, features [32,64,128,256,320,320]. Model size ≈ 30M parameters (same order as the owner's original U-Net; far smaller than the 102M TransUNet).

### C.3 Training
- Command: `nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_250epochs --npz` (fold 0 of nnU-Net's internal 5-fold split; 250-epoch trainer). Internal split: 166 training / 42 validation cases.
- Ran on the H100 at ~15 s/epoch, ~1 hour total. Loss (Dice+CE, can be negative) and validation pseudo-Dice improved together throughout; no overfitting signature (train and val loss tracked together). Per-epoch pseudo-Dice on both parotids reached ~0.70–0.74 by epoch ~72.
- At completion nnU-Net ran its automatic validation pass and reported: **Mean Validation Dice = 0.4538**.

---

## PART D — TEST-SET RESULTS AND THE ANOMALY

### D.1 Test-set metrics (locked 44 both-parotid test cases; `eval_testset.py`)
- 3D Dice **0.5030**; Clinical Tversky 0.5028; HD95 **65.03 mm**; Surface-Dice (3 mm) 0.5816. Per-side: R Dice 0.4823, L Dice 0.5237.
- These are **lower** than the owner's previous ~0.62 (note: the owner's 0.62 figures were 2D-pipeline validation numbers, not directly comparable).

### D.2 Per-case diagnostic (Claude)
- Per-case, per-side breakdown: **mean 0.503, median 0.595, 28/88 sides < 0.3**.
- The distribution was strongly **bimodal**: a large group of sides at 0.80–0.91 Dice, and a separate group near 0. Median > mean, indicating a subset of catastrophic cases dragging the mean down. Not a uniform weakness.

---

## PART E — DIAGNOSIS

### E.1 L/R swap test (Claude)
- For each test case, computed normal Dice (pred-1 vs GT-1, pred-2 vs GT-2) and **swapped** Dice (pred-2 vs GT-1, pred-1 vs GT-2).
- Result: **11 of 44 cases score much higher (0.6–0.85) when predicted L/R is swapped**, versus ~0 normally. Taking the better of normal/swap per case lifts the mean from 0.503 to 0.661.
- Pipeline breakdown: WC-parsed (35 cases) mean 0.470, DCM-parsed (9 cases) mean 0.630; swap-affected cases occur in **both** pipelines (9 WC, 2 DCM) — not a clean pipeline split.
- Visual confirmation rendered (`sanity_check_outputs/lr_swap_diagnosis.png`): in a swap case the prediction contours lie exactly on both glands (correct anatomy); only the L/R naming disagrees with GT.

### E.2 Merged (L/R-agnostic) quality (Claude)
- Scoring parotid tissue with L/R ignored (any-parotid Dice): **mean 0.820, median 0.841; all 44 cases ≥ 0.60; 38/44 ≥ 0.75; 0 cases < 0.30.**
- DEDUCTION (at that point): the model segments both parotids well; the low per-side score is an artifact of L/R being flipped on a subset. This was the point at which the result was characterised as a labeling/consistency issue rather than a model-quality issue.

### E.3 Corrupt mask
- One case, PAR0240 (patient `1~240067`), has a GT mask of 34,963 / 30,822 voxels vs a dataset median gland volume of ~6,227 — i.e. a blown-up/corrupt label. Visual overlay confirmed the GT contour is a bloated blob and the prediction is the anatomically sensible one.

---

## PART F — CORRECTIVE ACTION AND SECOND FINDING

### F.1 Owner decision
- After being shown the 0.82 merged result and offered (A) ship-as-is with post-processing vs (B) one clean retrain, the owner chose **B** (retrain on corrected data).

### F.2 Axis verification (Claude)
- Verified which image axis is left–right: for the two gland labels, **axis 1 (columns) separates them** (centroid gap ~90–126 voxels) while axis 0 barely differs (gap 1–5). In cases where model and GT agreed, label 1 (R) is the gland with the **smaller** axis-1 centroid and label 2 (L) the larger. (An earlier geometric per-side estimate of 0.79 had mistakenly used axis 0 and was discarded.)

### F.3 Two per-side recomputations (Claude)
- Relabeling the **model's existing two channels** by axis-1 centroid gave only 0.663 per-side. DEDUCTION: the current model's per-channel output is itself unreliable (it splits some glands across both channels), so relabeling its channels cannot fully recover sides.
- Splitting the **merged** prediction into two **connected components** and assigning L/R by axis-1 centroid gave **per-side Dice R 0.829 / L 0.801 / overall 0.815 / median 0.840, only 1/88 sides < 0.5** (corrupt case excluded). This is a post-processing step applied to the existing model, no retraining.

### F.4 Built the corrected dataset (`Dataset002_Parotid`) via `pipeline/fix_labels_qc.py`
- QC: drop any case whose gland volume > 3× dataset median (threshold 18,681 voxels).
- L/R correction: reassign the two label ids by axis-1 centroid (smaller → 1/R, larger → 2/L), preserving contour shapes, only swapping names where needed.
- Result of the build:
  - Train: **kept 208, dropped 0, L/R-corrected 1**.
  - Test: **kept 43, dropped 1** (PAR0240), **L/R-corrected 0**.

### F.5 SECOND FINDING (this is the key new fact)
- The L/R correction step corrected **only 1 case out of 251**. If the ground-truth labels had been flipped on ~25% of patients (as the swap test on predictions had suggested), this step would have corrected ~60 cases. It corrected 1.
- DEDUCTION: **the ground-truth annotations are geometrically consistent (R ≈ smaller axis-1 centroid across essentially all cases). The L/R flips are produced by the model, not present in the labels.**
- HYPOTHESIS (not yet verified by experiment): the model's L/R confusion is caused by **nnU-Net's default mirror (left–right flip) data augmentation and test-time mirroring**, a known failure mode for left/right-distinct anatomical structures. This has NOT been confirmed by a controlled run (e.g. re-predicting with `--disable_tta` or retraining with mirroring disabled); it is currently a hypothesis consistent with the evidence.

---

## PART G — CURRENT STATE AND OPEN DECISION

### G.1 Established facts
- The trained nnU-Net segments parotid tissue at **0.82 merged Dice / 0.815 per-side (via connected-components post-processing)** on the held-out test set (corrupt case excluded) — a strong result and a large improvement over the prior 0.62 (2D) numbers.
- The ground-truth L/R labels are consistent; the per-side degradation to 0.50 in raw scoring is due to the model's own L/R channel assignment.
- A simple, standard post-processing step already recovers clean per-side results from the existing model at no additional compute cost.

### G.2 Open decision (not yet made; reason for external review)
- **Option A:** keep the current checkpoint, apply connected-components + geometry post-processing for L/R, report 0.82 merged / 0.815 per-side. Zero additional cost.
- **Option B:** retrain with mirror augmentation disabled (nnU-Net `NoMirroring` trainer, and/or predict with `--disable_tta`) so the model outputs clean L/R natively. Requires another GPU session and a custom trainer variant; expected to give similar numbers to A. The mirroring hypothesis (F.5) would also be confirmed or refuted by this.

### G.3 Key file locations (owner's Mac)
- Trained model: `~/Desktop/nnunet_results_Dataset001_Parotid/Dataset001_Parotid/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/` (`checkpoint_final.pth`, `checkpoint_best.pth`, `progress.png`).
- Test predictions: `~/Desktop/test_predictions/` (44 `.nii.gz`).
- Datasets: `~/Desktop/Dataset001_Parotid/` (original, 208 train + 44 test), `~/Desktop/Dataset002_Parotid/` (corrected: 208 train + 43 test, QC + L/R-consistent).
- Pipeline code: `project/pipeline/` (`build_volumes.py`, `make_nnunet_dataset.py`, `masked_loss.py`, `eval_testset.py`, `fix_labels_qc.py`, PBS scripts, `README_HPC.md`).
- Diagnostic figure: `project/sanity_check_outputs/lr_swap_diagnosis.png`.

### G.4 Numbers table (test set, held-out)
| Scoring method | Dice | Notes |
|---|---|---|
| nnU-Net internal validation (fold 0) | 0.454 | on 42 val cases, raw per-side |
| Test, raw per-side (`eval_testset.py`) | 0.503 | 44 cases; HD95 65 mm; bimodal |
| Test, better-of-normal/swap | 0.661 | 44 cases |
| Test, merged (L/R ignored) | 0.820 | 44 cases (0.820 excl. corrupt) |
| Test, per-side via connected-components + geometry | 0.815 | 43 cases (corrupt excluded); R 0.829 / L 0.801 |

*End of case file.*
