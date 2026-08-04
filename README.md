# Parotid Auto-Segmentation on Head & Neck CT — and a Controlled Ablation That Corrected Its Own Headline

**Automated organ-at-risk (OAR) segmentation for radiotherapy planning, on an 844-patient Indian head-and-neck CT cohort — plus an 8-experiment ablation study showing that label quality, not model architecture, is what actually drives performance.**

> **The contribution is the study, not the score.** "I got 0.82 Dice" is a result. "I interrogated my own 0.82, found that ~62% of the apparent improvement was a measurement artifact, and identified what's really behind the rest" is research.

---

## How to read this repo

| If you have | Read |
|---|---|
| **60 seconds** | The TL;DR below, then *A retracted claim, kept on the record* |
| **10 minutes** | §5 (the L/R mirror bug — the best debugging story here) and §7 (the annotation gap — the best finding) |
| **You evaluate methods** | §6 (the ablation design) and §9 (limitations) |
| **You care about data engineering** | §3 (reverse-engineering an undocumented vendor format) |
| **You want to run the model** | §11 (weights) — and read the `--disable_tta` warning first |

---

## TL;DR

| | |
|---|---|
| **Task** | Bilateral parotid gland segmentation from planning CT (radiotherapy decision-support) |
| **Data** | 914 unique patients → **844 confirmed H&N** · **126,879 annotated CT slices** · single clinical institution |
| **Hard part #1** | 797 patients existed only in Elekta Monaco's **undocumented proprietary `.WC` contour format** — reverse-engineered from first principles, including the mm→pixel affine |
| **Hard part #2** | **51.6%** of parotid-bearing patients are contoured on **one side only** — a deliberate clinical decision that silently corrupts both training *and* evaluation |
| **Best model** | nnU-Net v2 `3d_fullres`, no-mirror trainer, **single fold** — **Dice 0.8187**, HD95 **5.25 mm**, Surface-Dice@3mm **0.8965** |
| **Scored on** | **43 both-parotid QC-clean cases** from the locked 165-patient test split, held completely unevaluated until the final run (+ **58 single-side** cases evaluated separately, §7) |
| **Honest gain** | **0.7434 → 0.8187 (+0.0753)** over the best from-scratch 2D baseline, *measured on the same locked test set* |
| **Headline finding** | Label quality dominates architecture. Six 2D models span **0.028** Dice; ImageNet pretraining buys **+0.006**; 5-fold ensembling is **not significant** (p=0.35) |
| **Debugging story** | nnU-Net's default L/R mirror augmentation made the model confuse the two glands — Dice 0.50, HD95 **65 mm**. Diagnosed with a controlled swap-test, fixed → 0.82 / 5.2 mm |

**Read the sample size before the score.** n=43 is small. §9 states the honest noise floor and which of these deltas survive it.

---

## ⚠️ A retracted claim, kept on the record

An earlier version of this README (and of my own résumé) quoted **"0.62 → 0.82 Dice."** **That comparison is retired and should never be cited again.**

The 0.62 was a *validation* number; the 0.82 a *test* number. When the four Phase-1 checkpoints were re-scored on the **same locked test set** (experiment **P0**), the identical weights scored **+0.1325 higher on average**:

| Model | Validation (original) | Locked test set (P0) | Δ |
|---|---|---|---|
| U-Net | 0.6209 | 0.7390 | +0.1181 |
| Attention U-Net | 0.6346 | **0.7434** | +0.1088 |
| TransUNet | 0.6332 | 0.7313 | +0.0981 |
| Swin-UNet | 0.5106 | 0.7156 | **+0.2050** |
| **mean** | 0.5998 | 0.7323 | **+0.1325** |

**Why:** the validation set contained single-side (partially-annotated) patients; the locked test set is both-parotid and QC-clean. A model that correctly predicts a gland the clinician deliberately chose not to contour is scored as wrong. **The 0.62 was deflated by the annotation gap on the evaluation side** — the same effect this study set out to measure, hiding inside its own baseline.

**The defensible statement is `0.7434 → 0.8187 (+0.0753)` on the same held-out test set.** This repo is written around that number.

---

## 1. The clinical problem

The partner institution treats **~800–1,000 head & neck cancer patients per year** with radiotherapy. Before treatment, an oncologist must manually contour every organ-at-risk on every CT slice — roughly **30 minutes per patient across ~10 organs**, done entirely by hand, with **no auto-contouring software in the department**.

The **parotid glands** are the primary target of this project:

- The tumour needs **≥60 Gy**; the parotids tolerate only **20–26 Gy**.
- Over-dosing them causes **xerostomia** — permanent dry mouth — leading to dental caries, mandibular necrosis, oral ulceration, and impaired swallowing and speech. It is largely **avoidable** with accurate contours.
- IMRT routes beams *around* the parotids, so contour accuracy is a **direct patient-safety requirement**, not a convenience.

This is **decision-support** — it assists and accelerates contouring. It is not an autonomous system and not a regulatory-cleared medical device.

---

## 2. Dataset

| Property | Value |
|---|---|
| Raw patient folders collected | 1,064 |
| Unique patients after merge/dedup | **914** (zero duplicates verified) |
| Confirmed H&N patients | **844** (70 non-H&N excluded) |
| Corrupt (mislabeled masks) excluded | 12 |
| Annotated CT slices | **126,879** |
| On-disk processed `.npz` | ~55 GB (consolidated to a 38 GB HDF5 for training) |
| Source | Single clinical institution (Elekta Monaco TPS, Siemens CT, 2 annotating radiation oncologists) |
| Locked split (patient-wise, `seed=42`) | **train 583 / val 84 / test 165** |
| **Evaluated test subsets** | **43 both-parotid QC-clean** (primary) · **58 single-side** (§7) |
| In-plane resolution | 512×512, 0.977 × 0.977 mm (90.5% of scans; range 0.643–0.977) |
| Slice thickness | 1.0 mm (H&N) or 3.0 mm |
| HU conversion | `HU = raw_pixel − 8192` (RescaleIntercept from DICOM) |

**The dataset is NOT released.** It is retrospective clinical data and is not publicly shareable. No patient-identifiable information exists in any processed file; all case identifiers in this repo are anonymised (`PAR0228`-style). Everything published here — code, configs, split logic, metrics, figures — is derived, not raw. See **Ethics & data use** at the end for the governance position, stated plainly.

**On demographic diversity:** virtually every published H&N OAR segmentation benchmark is trained and validated on Western (US/European) cohorts. This is an Indian-population dataset, which is a genuinely under-represented cohort in this literature.

---

## 3. Data engineering (the part that took the longest)

### 3.1 Reverse-engineering Elekta Monaco `.WC`

797 of the 914 patients existed **only** in Elekta Monaco's proprietary `.WC` contour format. There is **no public documentation, no specification, no open-source parser, and no GitHub repo** for it anywhere. It was decoded from scratch by binary/text inspection, cross-patient pattern matching, and iterative hypothesis testing against DICOM ground truth.

What it turned out to be: ASCII plaintext, **one file per CT slice** (~153 per patient), with the Z position encoded in the filename. From line 7 onward it is repeating blocks of `[n_points] / [organ_id] / [x y x y …]` — physical millimetre polygon coordinates. Integer organ IDs map to names through a per-patient `contournames` dictionary file. Spatial metadata (origin, spacing) has to be borrowed from any DICOM in the patient folder.

The coordinate transform, derived by hand:

```
Px = ( X - Ox) / Sx
Py = (-Y - Oy) / Sy
```

The **Y negation** is the non-obvious part: the source system's physical anterior–posterior axis is inverted relative to the image array's top-down axis. Get it wrong and every contour is vertically mirrored — which looks plausible enough on a single slice to slip through. Polygons are then rasterised with `cv2.fillPoly`.

### 3.2 DICOM-RTSTRUCT pipeline

The remaining 117 patients arrived as standard DICOM RTSTRUCT, parsed with `pydicom` — coordinates are already transformed by Monaco, so it is a matter of matching contour Z to slice `ImagePositionPatient[2]` and rasterising. Includes fallbacks for slices with missing/implicit transfer syntax and missing pixel data.

Both pipelines converge on **the same per-slice `.npz` schema**, so everything downstream is format-agnostic.

### 3.3 Consolidation, QC, and the locked split

- **Label standardisation** across all 126,879 files (parallel, 8 cores): canonicalises name variants (`RT_PAROTID / RIGHT_PAROTID / RT PAROTID → PAROTID_R`), drops TPS artifacts (`Foam_Core`, `Carbon_Fiber`, …), merges colliding labels by logical OR. **115,831 files modified, 0 errors.**
- **Statistical anomaly detection** on mask pixel distributions found **12 patients** whose "parotid" masks covered the entire head (20,000+ px/slice vs a normal 500–5,000). Excluded at load time; the underlying database was not altered. In Phase 2 this was formalised as an automated QC rule: drop any gland >3× the dataset median volume.
- **Patient-wise split locked with `seed=42` before any training** — all slices of a patient stay in one split, so there is no leakage. The **test set was held completely unevaluated** until the final nnU-Net phase.
- **Performance:** the naive sequential extraction managed ~340 patients in 9 hours before RAM saturation. A `ProcessPoolExecutor` rewrite (8 cores, streaming to disk) did the remaining **724 patients in under 2 hours**.
- **I/O:** 126,879 loose `.npz` files were an I/O disaster on the HPC filesystem. Packing them into a single 38 GB HDF5 gave **~10× faster loading**.

### 3.4 Dataloader

- HU windowing **−150 to +250** applied at load time (not baked into stored files, so the window stays a tunable).
- Min-max normalise to [0, 1] after windowing.
- Class imbalance handled with `WeightedRandomSampler`, **20× up-weighting on parotid-bearing slices** (slice-level, on the HDF5 loader that was actually used for HPC training; the local `.npz` loader uses 14× patient-level — the HPC value is the one behind every reported number).
- **Augmentation: horizontal flip only** — and when it fires, the **R and L mask channels are swapped**. Axial CT has valid left-right symmetry, but a naive flip silently mislabels which gland is which. Vertical flips and 90° rotations are excluded: the spine is always at the bottom, and inverting that violates the spatial prior.
- Loss: `0.5 × Dice + 0.5 × BCE`, **sigmoid not softmax** — left and right parotid are two independent binary problems, not mutually exclusive classes.

---

## 4. Phase 1 — four architectures, hand-implemented, controlled comparison

Four architectures written **from scratch in PyTorch** (no MONAI, no segmentation frameworks), each verified against its source paper, all trained under **identical conditions**: same loss, same optimiser (Adam, lr 1e-4), same schedule, same locked split, same sampler, no pretrained weights. The only variable is the architecture.

| Architecture | Params | Type | Note |
|---|---|---|---|
| U-Net (Ronneberger 2015) | ~31 M | CNN | baseline (ran without AMP; the other three used AMP) |
| Attention U-Net (Oktay 2018) | ~31.4 M | CNN + attention gates | gates on all 4 skips |
| TransUNet (Chen 2021) | ~102.5 M | Hybrid CNN–Transformer | ResNet-50 encoder (stage-3 = 6 blocks in code) + 12-layer ViT + CUP decoder |
| Swin-UNet (Cao 2021) | ~27 M | Pure transformer | shifted-window attention, zero conv in backbone |
| nnU-Net v2 | ~30 M | Self-configuring | Phase 2 (§5) |

**Results on the locked test set** (experiment P0 — 3D volumetric, same 43 cases as everything else):

| Model | 3D Dice | Tversky | HD95 (mm)† | Surface-Dice@3mm |
|---|---|---|---|---|
| **Attention U-Net** | **0.7434** | 0.7387 | 3.57 | 0.8827 |
| U-Net | 0.7390 | 0.7315 | 4.25 | 0.8702 |
| TransUNet | 0.7313 | 0.7307 | 5.60 | 0.8594 |
| Swin-UNet | 0.7156 | 0.7082 | 8.34 | 0.8794 |

† Phase-1 HD95 uses the isotropic 0.977 mm convention and is **not** comparable to the anisotropic nnU-Net numbers in §5. Dice and Tversky are directly comparable.

**The finding that mattered:** all four cluster within **0.028 Dice** across a 31M → 102M parameter range. That is not a capacity problem — it is a **ceiling imposed by the data and preprocessing**, which is what motivated Phase 2 and, eventually, the whole ablation study.

**Evaluation metrics** (used consistently throughout): 3D volumetric Dice · **Clinical Tversky (α=0.3, β=0.7)**, asymmetric to penalise under-segmentation more heavily because in radiotherapy a missed gland receives radiation the plan assumed it wouldn't · **HD95** (95th-percentile Hausdorff, mm) · **Surface Dice @3mm tolerance**.

---

## 5. Phase 2 — nnU-Net, and the left/right mirror bug

Phase 2 moved to **nnU-Net v2** (`3d_fullres`, from scratch), which auto-configured to spacing `[3.0, 0.977, 0.977]` (anisotropy preserved, not forced isotropic), patch `[48, 224, 192]`, batch 2, CTNormalization, 6 stages, features `[32,64,128,256,320,320]` — about 30M params, the same order as the hand-built U-Net and a third the size of TransUNet.

### The bug

**The first nnU-Net run scored 0.5030 Dice with HD95 65.03 mm — worse than every baseline.** Training looked perfectly healthy: train and val loss tracked together, pseudo-Dice ~0.70–0.74. That mismatch is the interesting part.

The diagnostic chain:

1. **Per-case Dice was strongly bimodal** — a cluster at 0.80–0.91 and a separate cluster near zero. Median (0.595) > mean (0.503): a subset of catastrophic failures, not uniform weakness. 28 of 88 sides scored below 0.3.
2. **Swap test:** score each case normally *and* with predictions swapped (pred-R vs GT-L). **11 of 44 cases scored 0.6–0.85 when swapped versus ~0 normally.** Affected cases came from *both* parsing pipelines (9 WC, 2 DICOM) — so not a parser bug.
3. **L/R-agnostic Dice** (score parotid tissue, ignore which side): **mean 0.820, all 44 cases ≥ 0.60, none below 0.30.** So the model finds both glands correctly; only the *naming* is wrong on a subset.
4. **Ruled out flipped ground truth:** a geometric relabelling pass (assign R/L by column centroid) corrected **1 of 251 cases**. If a quarter of the GT were flipped it would have corrected ~60. The flips are produced by the model, not present in the data.
5. **Root cause: nnU-Net's default left-right mirror augmentation** (plus test-time mirroring) makes L/R-distinct structures interchangeable. Isolating it confirmed the diagnosis: `--disable_tta` *alone* recovered only 0.503 → 0.564 with HD95 still ~64 mm, proving the confusion was **baked into the weights**, not just inference-time.
6. **Fix:** retrain with a custom `nnUNetTrainer_250epochs_noMirror` (mirroring disabled) and predict with `--disable_tta`.

| Metric | Mirror ON (default) | Mirror OFF (fixed) |
|---|---|---|
| nnU-Net internal fold-0 val Dice | 0.4538 | **0.7967** |
| Test 3D Dice | 0.5030 | **0.8187** |
| HD95 | 65.03 mm | **5.25 mm** |
| Surface Dice @3mm | 0.5816 | **0.8965** |

> **On novelty:** that nnU-Net's mirroring can confuse laterally paired structures is known in the community and appears in the framework's own issue tracker. What this repo contributes is the **decomposition** — the bimodal per-case distribution, the swap test, the L/R-agnostic Dice separating *detection* from *naming*, the geometric relabelling that rules out flipped ground truth, and the isolation showing `--disable_tta` alone recovers only to 0.564. The diagnosis is the contribution, not the observation.

### Final model

| Model | 3D Dice | Tversky | HD95 (mm) | Surf-Dice@3mm |
|---|---|---|---|---|
| **nnU-Net 3d_fullres, noMirror, single fold** ← *the deliverable* | **0.8187** | 0.8166 | **5.25** | 0.8965 |
| nnU-Net noMirror, 5-fold ensemble | 0.8202 | 0.8174 | 5.24 | 0.8990 |

Per-side: R 0.8278 / L 0.8095. **The 5-fold ensemble is not worth it** — see E5 below. One unseen real patient run end-to-end scored 0.866.

---

## 6. The ablation study — eight experiments, one locked test set

Rather than stop at 0.82, I ran a controlled study to find out *what actually caused it*. Rules: **one variable per experiment**, every model scored on the **same locked 43-case test set** through the same evaluator, hypotheses pre-registered before each GPU run.

### All results

| Experiment | Condition | Dice | Tversky | HD95 (mm) | Surf-Dice | Isolates |
|---|---|---|---|---|---|---|
| *(ref)* | nnU-Net 3d_fullres, 1 fold | **0.8187** | 0.8166 | 5.25 | 0.8965 | — |
| **E5** | nnU-Net 5-fold ensemble | 0.8202 | 0.8174 | 5.24 | 0.8990 | ensembling → **n.s. (p=0.35)** |
| **E1** | nnU-Net **2d**, 1 fold | 0.8117 | 0.8085 | 5.52 | 0.8943 | 2D vs 3D → **+0.007 only** |
| **E2** | 3D, **dirty labels** (430 pts) | 0.7726 | 0.7644 | 5.45 | 0.8400 | annotation gap, realistic → **−0.046** |
| **E2b** | 3D, **gapped labels, constant-N** (208) | 0.6899 | 0.6680 | 9.63 | 0.7649 | annotation gap, isolated → **−0.129** |
| **E4** | Hand-built plain 3D U-Net + largest-CC | 0.7750 | 0.7788 | 6.05 | 0.8539 | architecture → **−0.044 residual** |
| **E6** | Custom 3D U-Net, **masked loss** (430) + CC | 0.7589 | 0.7746 | 9.19 | 0.8403 | annotation-gap *fix* |
| **E7** | **Per-side specialist experts** (L320 + R318) | 0.8099 | 0.8096 | 5.85 | 0.8896 | annotation-gap *decompose* |
| **P0** | Attention U-Net (from scratch, 2D) | 0.7434 | 0.7387 | 3.57* | 0.8827 | Phase-1 baseline on test |
| **E3** | TransUNet, **ImageNet-pretrained** | 0.7373 | 0.7325 | 3.78* | 0.8759 | pretraining → **+0.006** |
| **E3** | Swin-UNet, ImageNet-pretrained | 0.7185 | 0.7122 | 6.70* | 0.8679 | pretraining → +0.003 |

`*` isotropic HD95 (Phase-1 protocol) — never compare these to the anisotropic rows.

Every row was independently re-verified against the raw per-case eval CSVs, not the writeups.

### What actually drives the +0.0753

| Axis | Δ Dice | Share | Experiment |
|---|---|---|---|
| **Label quality (net, as suffered)** | **+0.0461** | **~61%** | E2: dirty-430 0.7726 → clean 0.8187 |
| Preprocessing + nnU-Net training recipe | ~+0.0222 | ~29% | remainder |
| Dimensionality (2D → 3D) | +0.0070 | ~9% | E1 |
| Ensembling (1 → 5 folds) | +0.0015 (**n.s.**) | ~2% | E5 |

⚠️ **Treat the exact percentages as indicative, not measured.** The paired noise floor (§9) makes E2's net −0.046 borderline, and attribution is not strictly additive — E4 shows the training machinery alone (+0.044) exceeds the +0.022 "preprocessing" remainder. **The robust claim is the ordering, and it rests on E2b's latent −0.129 and the evaluation-side −0.133, both far above any noise floor.**

---

## 7. The star finding: the annotation gap

During review, models kept predicting **bilateral** parotids where the ground truth had only **unilateral** annotation. The obvious read is model hallucination. It isn't.

**The project's clinical advisor — a practising radiation oncologist — confirmed the predictions were anatomically correct.** When a tumour is strictly one-sided, clinicians **deliberately do not contour the contralateral healthy-side parotid** — it isn't at risk in that treatment plan, and skipping it saves ~15 minutes per structure. This is a **correct clinical decision, not an annotation error.**

It is also, quantitatively, the largest single factor in this entire project — and it hits **both sides of the ledger**:

| Where it hits | Effect | Experiment |
|---|---|---|
| **Training** (isolated, constant N) | **−0.1288** Dice · HD95 5.25 → 9.63 mm | E2b |
| **Training** (net, with 2× data compensating) | −0.0461 Dice | E2 |
| **Evaluation** (deflates validation scores) | **−0.1325** Dice on *identical weights* | P0 vs original val |
| Prevalence | **222 / 430 = 51.6%** of parotid-bearing train/val patients | E2 scan |
| Structure | the 222 extra partial patients buy back **64%** of the damage (+0.0827) | E2 − E2b |

Supporting signal: Tversky (β-weighted against false negatives) falls *further* than Dice in both damaged arms — exactly the signature of a model being taught to omit glands.

### Four ways to handle it — all four tested

| Strategy | What it does | Both-parotid (43) | Single-side (58)‡ |
|---|---|---|---|
| **Discard** (use only clean both-annotated) | drop 222 patients | **0.8187** | 0.8523 |
| **Decompose** (per-side specialist experts, E7) | two single-class models | 0.8099 | **0.8552** |
| **Mask** (masked partial-label loss, E6) | exclude un-annotated channel from loss | 0.7589 | 0.8010 |
| **Penalise** (naive: train on everything as-is) | the default, wrong thing | 0.7296 | 0.7436 |

‡ 58 held-out single-side patients, scoring the annotated (at-risk) gland. This set was added specifically to close the study's biggest self-identified gap — the 43-case set is the "easy subset" and structurally cannot reward the masking strategy's main benefit.

**Two conclusions:**

1. **On this cohort, no strategy for exploiting single-side data beats simply discarding it.** Data *quality* > data *quantity*.
2. **Single-side is not a failure mode.** The at-risk gland scores **~0.85** — *higher* than the both-side average — so real-world performance on the gland that clinically matters is good.

**A prediction I made, tested, and got wrong:** the ablation evidence led me to predict that a masked partial-label loss would capture all 430 patients' data benefit without the label penalty and would **beat 0.8187**. E6 tested it. Masking **does** beat the naive control (+0.029) and trains far more stably (the ordinary-loss control diverged), but it reached only 0.7589 — below the clean baseline. **The prediction was refuted.** Making a number-backed prediction, testing it, and reporting the miss is a stronger outcome than never checking.

---

## 8. Architecture is near-irrelevant — five independent lines of evidence

1. **Six 2D models** (4 from scratch + 2 ImageNet-pretrained) span **0.0278 Dice** (0.7156 – 0.7434).
2. **E3 — pretraining doesn't rescue the transformers.** ImageNet init adds **+0.006** (TransUNet, a clean 258/258 exact weight load into an identical 102.5M architecture) and **+0.003** (Swin). The best pretrained transformer (0.7373) **still loses to a from-scratch Attention U-Net (0.7434)**. Pretraining's real benefit is **boundaries** — HD95 improves ~1.8 mm on both arms. Mechanism: the pretrained TransUNet drove train loss to 0.0545, a third of its own val loss — it memorised the parotid slices without generalising. So the *dataset*, not the initialisation, is the ceiling.
3. **E4 — a plain hand-built 3D U-Net** (16.5M params, deliberately vanilla) on nnU-Net's preprocessing reaches 0.7681, and **one line of largest-connected-component postprocessing collapses its HD95 from 26.76 → 6.05 mm** (≈ nnU-Net's 5.25). **nnU-Net's boundary edge is postprocessing, not architecture.** The remaining −0.044 Dice traces to its training machinery (deep supervision, heavy augmentation), not the model.
4. **E1** — 2D vs 3D is worth +0.007.
5. **Phase 1** — 31M → 102M params produced no gain.

**A1 — why Swin-UNet's boundaries were bad** (a sub-study): two compounding causes, cleanly separated. **Stray false-positive islands dominate** (2.84 vs 1.20 connected components per gland, 75.6% vs 16.3% multi-component, strays up to 306 mm; removing them drops HD95 3.77 mm vs 0.04 mm for U-Net, paired Wilcoxon p ≈ 1.3e-11). **Coarse 128→512 upsampling is the residual floor** — even after island removal Swin sits ~2 mm above U-Net. Also: Swin's apparent "collapse" in the original validation numbers (0.5106) was **largely a val-set artifact** — on the locked test set it scores 0.7156, a +0.205 jump, the largest of any model.

---

## 9. Limitations (read these before citing anything)

- **n = 43** both-parotid test cases (+58 single-side). The honest cross-run **paired noise floor is ~0.05 Dice**. Measured paired-delta stds: E5 (same architecture, single-vs-ensemble) **0.017**; Attention-vs-U-Net **0.047**; genuinely different cross-run pairs plausibly ~0.05–0.08. Note the single-model per-case dispersion is ≈0.17 — case difficulty varies a lot, which is exactly why paired comparison is the right test.
- **Therefore:** E1 (+0.007), E3 (+0.006/+0.003) and E5 (+0.0015) are **"no measurable effect," not zero** — and E2's net −0.046 is **borderline**. The robust numbers are E2b's latent **−0.129** and the evaluation-side **−0.133**, both far above any noise floor.
- **Single institution, single fold, single seed.** No seed-variance estimate.
- **HD95 protocols differ:** Phase-1/E3/A1 use isotropic 0.977 mm; nnU-Net/E2/E4/E6/E7 use true anisotropic (0.977, 0.977, 3.0). **Never put them in one column.**
- **Attribution is not strictly additive** — E4 shows the training machinery alone (+0.044) exceeds the +0.022 "preprocessing" remainder.
- **Parotids only.** Multi-organ generalisation is unmeasured — though it has been scoped; see §10.
- **The 0.82 is an optimistic operating point** — the primary test set is the QC-clean both-parotid subset. Messy/QC-failing cases remain unmeasured.
- Claims are **specific to this dataset**. The lessons are offered as transferable hypotheses worth testing elsewhere, not as proven universal laws.

---

## 10. Multi-organ: scoped, not speculative

A 150-patient random audit of the processed dataset established which additional OARs are annotated densely enough to model. **The central constraint: a missing mask does not mean the organ is absent** — the same annotation gap as §7, generalised across structures.

| Structure | % of patients with ≥1 annotated slice | Note |
|---|---|---|
| EYE_L / EYE_R | 65% / 64% | good bilateral target |
| OPTIC_NERVE_L / _R | 65% / 15% | R badly under-annotated |
| TEMPORAL_LOBE | 61% | easy, optional |
| SPINAL_CORD_PRV / SPINAL_CORD | 54% / 41% | key OAR |
| LARYNX | 51% | good target |
| **PAROTID_L / PAROTID_R** | **46% / 45%** | **primary clinical target — this repo** |
| OPTIC_CHIASMA | 42% | tiny, hard |
| LENS_R / LENS_L | 59% / 15% | very small, hardest |
| BRAINSTEM | 5% | too rare to model reliably |

**Verdict: multi-organ is feasible** on a ~8-organ target set (parotids, eyes, spinal cord, larynx, optic nerves, optionally optic chiasm and temporal lobe). The audit's endorsed approach: resample to common spacing + intensity-normalise + augment; masked/marginal loss for the partial labels; keep the volume-based QC filter; and **run the locked test set once, at the very end.**

Note the honest tension with §7: masking *underperformed* discarding on parotids. Whether that holds when the partial-label structure is spread across eight organs rather than two sides is an open question — and the reason this is listed as scoped rather than solved.

---

## 11. Repository layout

```
pipeline/                      Phase-2 nnU-Net pipeline
  build_volumes.py               per-slice npz -> 3D NIfTI (Z from filenames, spacing assumptions documented)
  make_nnunet_dataset.py         nnU-Net raw dataset builder (0=bg, 1=parotid_r, 2=parotid_l)
  fix_labels_qc.py               geometric L/R consistency + >3x-median volume QC filter
  eval_testset.py                THE evaluator - Dice / Tversky / HD95 / Surface-Dice, true anisotropic spacing
  masked_loss.py                 masked partial-label loss (+ unit test) for the annotation gap
  nnUNetTrainer_250epochs_noMirror.py   the custom trainer that fixed the L/R bug
  test_and_visualize.py          inference + axial montages + 3D surface renders + HTML gallery
  README_HPC.md / README_CLOUD_*.md     reproducible runbooks (PBS + RunPod)

phase1/                        Phase-1 from-scratch models & data engineering
  parsers/                       .WC reverse-engineered parser family + DICOM-RTSTRUCT parsers
  preprocessing/                 label standardisation, splits (seed 42), QC/counting utilities
  dataloaders/                   npz + HDF5 loaders, HU windowing, weighted sampling, L/R-aware flip
  models/                        unet.py, attention_unet.py, transunet.py, swin_unet.py (+ train/eval/PBS)
  loss_function.py               0.5*Dice + 0.5*BCE, sigmoid, per-channel

ablation/                      the eight-experiment controlled study
  ABLATION_PLAN.md               design, pre-registered hypotheses, one-variable rule
  SYNTHESIS.md                   the full write-up (this README is its summary)
  E1..E7, P0, A1/                per-experiment code + RESULT.md + eval CSVs
  results/                       every eval.csv and per-case CSV

results/                       metric CSVs, training histories, learning curves
figures/                       prediction montages, comparison grids, diagnostic figures
docs/                          MASTER_PROJECT_REFERENCE.md, PROJECT_NARRATIVE.md, data audit,
                               the L/R-bug case file, POD_UPLOAD_PLAYBOOK.md
```

**Not in this repo:** patient data in any form (raw, processed, or NIfTI), the 38 GB HDF5, and nnU-Net preprocessed caches. Model weights are attached to the GitHub **Release**, not tracked in git.

⚠️ **Some older scripts contain stale hardcoded paths** from earlier folder layouts. Check paths before running anything outside the documented runbooks.

---

## 12. Model weights

The final model is available under [**Releases → v1.0**](https://github.com/Ritvik-Mod/hn-oar-segmentation/releases/tag/v1.0) (219 MB) — nnU-Net `3d_fullres`, no-mirror trainer, fold 0, with `plans.json`, `dataset.json`, the training log and loss curve. Unzips straight into an `nnUNet_results` tree.

⚠️ **It must be run with `--disable_tta`.** This model was trained with mirror augmentation disabled; nnU-Net's default test-time mirroring reintroduces the exact left/right confusion described in §5 (Dice 0.82 → 0.50). Full usage instructions are in `README_WEIGHTS.md` inside the archive.

---

## 13. Reproducibility

The **data cannot be released**, so full end-to-end reproduction is not possible outside the source institution. What *is* verifiable here:

- Every parser, dataloader, loss, model, trainer, evaluator, and ablation script.
- The **exact evaluator** (`pipeline/eval_testset.py`) that produced every number in this README, with its validation cases: a perfect prediction gives Dice 1.0 / HD95 0; a deliberate 4-voxel shift gives Dice 0.68 / HD95 3.82 mm ≈ 4 × 0.977 mm, physically correct.
- The **locked split logic** and all split JSONs (patient IDs are anonymised, no PHI).
- Every **raw eval CSV and per-case CSV** behind every table above — the aggregates can be recomputed from them, and were, in an independent verification pass.
- The `masked_loss.py` unit test, which proves that corrupting an un-annotated channel leaves the loss bit-identical.
- Complete **runbooks** for the HPC (PBS) and RunPod paths, including the trainer-registration gotchas.

**Compute used:** university HPC (2× NVIDIA L40, PBS) for Phase 1; rented RunPod H100 / L40S / 4090 / A40 for Phase 2 and the entire ablation study. The whole eight-experiment ablation cost **under $30 of GPU time**.

**Tech stack:** Python 3.10+ · PyTorch 2.x · nnU-Net v2 · pydicom · h5py · NumPy/SciPy · OpenCV · scikit-image · nibabel · matplotlib · `concurrent.futures.ProcessPoolExecutor`. See `requirements.txt`.

---

## 14. Status

```
[x] .WC format reverse-engineered + parser                    complete
[x] DICOM-RTSTRUCT pipeline                                    complete
[x] Label standardisation, QC, locked split (seed 42)          complete
[x] HDF5 consolidation + weighted, L/R-aware dataloader        complete
[x] Four from-scratch architectures, controlled comparison     complete
[x] Clinical evaluation framework (Dice/Tversky/HD95/SurfD)    complete
[x] nnU-Net v2 3D pipeline + L/R mirror bug diagnosis & fix    complete
[x] 5-fold ensemble (measured; not significant - single fold shipped)
[x] Ablation study: E1 E2 E2b E3 E4 E5 E6 E7 P0 A1 + synthesis complete
[x] Single-side held-out evaluation (58 patients)              complete
[x] Independent verification of all results vs raw CSVs        complete
[x] Prediction gallery (15 models x 43 cases)                  complete
[ ] Multi-organ extension (~8 OARs, masked partial-label loss) scoped (§10)
[ ] Conference submission (ICCI 2026, extended abstract)       drafted
```

---

## Ethics & data use

Retrospective, fully de-identified clinical CT and radiotherapy planning data from a single institution, used with departmental permission. **Formal institutional ethics-board approval is not in place**, and this is stated rather than left to be assumed. No patient-identifiable information is present in any processed file or anywhere in this repository; all case identifiers are anonymised and the mapping is not published.

**The dataset is not shareable** — not on request, and not under a data-use agreement. Nothing in this repository is derived from raw patient data in a form that could be inverted.

This is a research and decision-support project — **not a medical device, not regulatory-cleared, and not for clinical use.**

---

## Citation

```bibtex
@misc{mod2026parotid,
  author = {Mod, Ritvik},
  title  = {Parotid Auto-Segmentation on an Indian Head-and-Neck CT Cohort:
            A Controlled Ablation Showing Label Quality Dominates Architecture},
  year   = {2026},
  note   = {https://github.com/Ritvik-Mod/hn-oar-segmentation}
}
```

## Author

**Ritvik Mod** — B.Tech Computer Science & Engineering, BIT Mesra (2024–2028)
[ritvikmod@gmail.com](mailto:ritvikmod@gmail.com) · [github.com/Ritvik-Mod](https://github.com/Ritvik-Mod)

**Clinical advisor:** Dr. Hemendra Mod, radiation oncologist (20+ years' practice), who provided the ground-truth clinical judgement — including the confirmation that turned an apparent model failure into this project's central finding. *Disclosure: the clinical advisor is a family member. The clinical claims he verified are checkable against the radiotherapy literature, and the annotation-gap finding is supported independently by the quantitative evidence in §7.*

Solo project: data engineering, reverse-engineering, model implementation, training, evaluation, and the ablation study.
