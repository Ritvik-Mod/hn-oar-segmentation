# PROJECT NARRATIVE & POSITIONING

> **What this is.** The single source of truth for how this project is *described* to the outside world —
> the conference abstract/talk, the résumé/CV brief, the README, and any future paper. Every outward-facing
> artifact should draw its framing and numbers from here so nothing drifts. The **technical** record is
> `MASTER_PROJECT_REFERENCE.md`; this is the **story** record.
>
> **Framing rule (important):** claims are **specific to this dataset** (one institution, parotids, n=43
> test); lessons are offered as **transferable hypotheses**, never as proven universal laws. Say *"on our
> data, X"* and *"this suggests, and is worth testing elsewhere, that Y."*
>
> **⚠️ This supersedes any copy that quotes "0.62 → 0.82."** That comparison is retired (see §Canonical
> numbers). Owner: Ritvik Mod. Updated 2026-07-17 after the ablation study.

---

## 1. THE ONE-LINE THESIS

> On a 914-patient Indian head-and-neck CT cohort, we built a parotid auto-segmentation model **and then ran
> a controlled ablation that dissected its own headline result** — finding that most of the apparent
> improvement was a measurement artifact, and that **label quality, not model architecture, is what actually
> drives performance** on partial-label clinical data.

The contribution is **the study, not the score.** "I got 0.82 Dice" is a result; "I interrogated my own 0.82
and found what's really behind it" is a piece of research.

## 2. THE ARC (how to tell it)

1. **The problem is real.** Parotid glands are a critical organ-at-risk in H&N radiotherapy; manual
   contouring is a ~30-min/patient bottleneck at a centre treating 800–1,000 patients/year.
2. **We built the pipeline** — from reverse-engineering the hospital's proprietary contour format to a 3D
   nnU-Net, reaching **0.82 Dice** on a held-out test set. (Strong, but this is the *setup*, not the point.)
3. **We didn't trust our own headline.** A controlled 8-experiment ablation on a single locked test set
   asked: what actually caused the gain — 3D, preprocessing, labels, ensembling, architecture, pretraining?
4. **The correction.** ~62% of the famous improvement was a **validation-vs-test measurement artifact.** The
   honest gain is **0.743 → 0.819 (+0.075)** on the same test set.
5. **The real driver is the data.** Of what remains, **label quality dominates**; architecture, 3D,
   pretraining and ensembling are near-irrelevant. The **annotation gap** — clinicians deliberately leaving
   the healthy-side parotid un-contoured — is the single largest quantitative factor, and it hits *both* the
   training side and the evaluation side.
6. **We tested the forward path — and it didn't pan out as predicted, which is the rigorous outcome.** We
   built the masked partial-label loss and ran it (E6): masking the un-annotated gland **beats the naive
   all-data control (+0.029 Dice) and trains far more stably**, but it **did not beat the clean both-parotid
   baseline (0.759 vs 0.775) or nnU-Net (0.819)** on this test set — which, being all both-parotid, can't
   reward masking's main benefit anyway. We made a number-backed prediction, tested it, and it half-held; the
   proper test is a single-side evaluation. (Predicting, testing, and reporting a miss is *stronger* science
   than never checking.)

## 3. CANONICAL NUMBERS (use these exact figures everywhere; never the retired ones)

**Dataset:** 914 patients (844 H&N), **126,879** annotated CT slices, single Indian institution (Elekta
Monaco TPS, Siemens CT, 2 oncologists). Proprietary `.WC` contour format reverse-engineered from scratch +
DICOM-RTSTRUCT. Locked patient-wise split 583/84/165 (seed 42).

**Honest headline (all on the SAME locked 43-case test set):**
- Best from-scratch 2D baseline (Attention U-Net) **0.7434** → nnU-Net 3D single fold **0.8187** = **+0.075**.
- nnU-Net 5-fold ensemble 0.8202 (**not** significantly better than single fold; p≈0.35 — ship the single fold).

**The correction:** ~**62%** of the previously-quoted "0.62 → 0.82 (+0.20)" was a val-vs-test artifact — the
0.62 was a *validation* number deflated by the annotation gap; the 0.82 a clean *test* number. **Never quote
0.62 → 0.82 again.** The defensible statement is **"0.743 → 0.819 (+0.075) on the same held-out test set."**

**What drives the real +0.075 (controlled A/B on the locked test set):**
| Axis | Δ Dice | Note |
|---|---|---|
| **Label quality** | **+0.046 net / +0.129 latent** | dominant driver |
| Preprocessing + nnU-Net recipe | ~+0.022 | (bundles training machinery) |
| Dimensionality (2D→3D) | +0.007 | within noise |
| Ensembling (1→5 fold) | +0.0015 | not significant |

**Architecture is near-irrelevant:** six 2D models span **0.028** Dice; ImageNet pretraining adds **+0.006**
(TransUNet) / +0.003 (Swin) and both still lose to a from-scratch CNN; a hand-built 3D U-Net + trivial
connected-component postprocessing **matches nnU-Net's boundary accuracy** (HD95 ≈ 5–6 mm).

**The annotation gap (the star finding):** clinicians deliberately skip the healthy contralateral parotid on
one-sided tumours — **51.6%** of parotid-bearing patients are single-side. It costs **~0.13 Dice in
training** *and* **deflates validation scores by ~0.13** on identical weights. Clinically validated by Dr.
the project's radiation-oncologist advisor. This is the part genuinely novel relative to the published literature.

**Attempted fix — masked loss (E6, tested):** training on all 430 patients with the un-annotated gland masked
out of the loss **beats the naive all-data control (+0.029 Dice) and stabilises training** (the ordinary-loss
control diverged), but on this both-parotid test set it did **not** beat the clean-208 baseline (0.759 vs
0.775) or nnU-Net (0.819). The §21 prediction that masking would beat the best was **not confirmed**; the
proper test — a held-out single-side evaluation — we then ran (next).

**Single-side evaluation + per-side experts (E7, done):** we then scored every model on the **58 held-out
single-side test patients** the study had never measured (leakage-clean). Two findings: (1) **the at-risk
gland scores ~0.85 on single-side cases — *higher* than the both-side average (~0.82)** — so single-side is
**not** the feared failure mode; real-world performance on the clinically-relevant gland is good, which
resolves the study's biggest caveat. (2) **Per-side specialist models** (a left-parotid expert on 320
patients + a right on 318 — the "decompose" treatment, **not** a mixture-of-experts) are **competitive but
don't win overall**: both-parotid 0.810 (< nnU-Net 0.819), single-side 0.8552 (≈ nnU-Net 0.8523, within
noise). This **completes the four-way annotation-gap treatment** — discard (0.819, best) / decompose (0.810)
/ mask (0.759) / penalise (worst) — and its lesson reinforces the headline: **clean-label data quality beats
data quantity**; adding single-side patients (any strategy) doesn't beat simply using the clean both-parotid
set. Contralateral suppression (the gap's fingerprint, §16) confirmed but modest: the gap-suffering model
predicts the healthy gland in 88% of cases vs 95–100% for the others.

**Bonus story (keep it — it's a great debugging narrative):** nnU-Net's default left–right mirror
augmentation made the model confuse the two glands (test Dice 0.50, HD95 65 mm); diagnosed via a controlled
swap-test, fixed with a no-mirror trainer → 0.82 / 5.2 mm.

**Noise floor / honesty:** n=43; cross-run paired deltas below ~0.02–0.05 are within noise (so the 3D,
pretraining, and ensembling effects are reported as "no measurable effect," not zero). Single fold/seed. The
primary test set is the **both-parotid QC-clean subset** (0.82). Single-side performance was **separately
measured** in E7 on 58 held-out patients (at-risk gland ~0.85) — so the earlier "we never measure single-side"
caveat is now **resolved**; what remains unmeasured is messy/QC-failing cases and multi-organ generalisation.

## 4. THE FOUR KEY MESSAGES (what a listener should remember)

1. **The data is the product, not the model.** On our cohort, label quality and preprocessing beat every
   architectural choice by ~20×. Effort on annotation quality pays far more than effort on architectures.
2. **Evaluate honestly or your own metrics will lie to you.** A val-vs-test mismatch inflated our headline by
   ~62%; we caught it only by re-scoring on one locked test set.
3. **The annotation gap is a first-class effect, not a footnote.** A deliberate, clinically-correct labelling
   choice is the dominant quantitative factor on both sides of the ledger.
4. **Rigor over trophies.** The value here is a controlled study that corrected its own result — a
   transferable cautionary tale for anyone training on retrospective clinical labels.

## 5. WHAT'S GENUINELY NOVEL (defensible)

- A quantified, clinically-validated characterisation of the **annotation gap** and its **two-sided**
  (training + evaluation) cost on a real cohort.
- A rare **Indian-demographic** H&N dataset assembled by **reverse-engineering a proprietary contour format**.
- A **controlled decomposition** attributing a segmentation improvement across six confounded axes on one
  locked test set — including a self-caught measurement artifact.
- A **four-way empirical treatment of the annotation gap** — discard / penalise / mask / decompose — showing
  that on this cohort **no strategy for using single-side data beats simply using the clean both-parotid set**
  (data quality > quantity), plus a held-out single-side evaluation confirming real-world at-risk-gland
  performance is strong (~0.85).

## 6. SCOPE / DISCLAIMER (one line to include everywhere)

*Single-institution, parotids only; n=43 both-parotid + 58 single-side held-out test; single fold/seed. Findings
are a case study; the transferable lessons are hypotheses for other cohorts. Decision-support, not a medical
device.*

---

## 7. READY-TO-USE COPY (paste and trim; do not re-derive numbers)

### 7a. One-liner (résumé header / LinkedIn)
Built a parotid auto-segmentation pipeline on a 914-patient Indian head-and-neck CT cohort and ran a
controlled ablation showing **label quality — not architecture — drives performance**, after catching a
val-vs-test artifact that had inflated the headline by ~62%.

### 7b. Two-sentence (README intro)
This project auto-segments the parotid glands on head-and-neck CT for radiotherapy planning, reaching 0.82
Dice on a locked test set from a 914-patient Indian cohort. A controlled 8-experiment ablation then shows the
real driver is label quality — specifically a clinically-deliberate "annotation gap" — not model
architecture, and corrects a validation-vs-test measurement artifact in the original headline.

### 7c. CV / résumé bullets
- Built an end-to-end 3D parotid-segmentation pipeline (proprietary-format reverse-engineering → nnU-Net) on
  a **914-patient, ~127k-slice** Indian head-and-neck CT dataset; **0.82 Dice on a held-out test set**.
- Ran an **8-experiment controlled ablation** that **caught a validation-vs-test measurement artifact
  inflating the reported gain by ~62%**, and attributed the true improvement across six confounded axes.
- Showed **label quality dominates architecture** (six models within 0.028 Dice; ImageNet pretraining +0.006;
  ensembling not significant); quantified a clinically-validated **annotation gap** costing ~0.13 Dice on
  both the training and evaluation sides.
- Diagnosed and fixed a left/right label-confusion failure from nnU-Net mirror augmentation via a controlled
  swap-test (test Dice 0.50 → 0.82, HD95 65 → 5 mm).

### 7d. Abstract-length (~190 words) — the seed for the conference abstract
We study automated parotid-gland segmentation on 914 head-and-neck CTs from an Indian hospital cohort — a
population under-represented in published literature — assembled by reverse-engineering the hospital's
proprietary contour format. A 3D nnU-Net reaches 0.82 Dice on a locked held-out test set. Rather than stop at
the score, we run an eight-experiment controlled ablation on a single locked test set to explain it.
Re-scoring the original models on the test set shows **~62% of the reported improvement was a
validation-vs-test measurement artifact**: the honest gain is 0.743 → 0.819 (+0.075). Decomposing the
remainder, **label quality is the dominant driver** (+0.046 net, +0.129 when isolated), ahead of
preprocessing (+0.022), dimensionality (+0.007) and ensembling (+0.0015, not significant). Architecture and
initialisation are near-irrelevant: six models span 0.028 Dice and ImageNet pretraining adds +0.006. The
single largest effect is a clinically-validated **annotation gap** — clinicians deliberately leave the
healthy-side parotid un-contoured in 51.6% of cases — which costs ~0.13 Dice in training and deflates
validation by ~0.13. On our cohort the data, not the model, is the product; we offer this as a transferable
caution for partial-label clinical segmentation.

### 7e. Talk arc (slide beats, ~10 min)
1. The clinical stakes (parotids, xerostomia, the contouring bottleneck).
2. The dataset & the reverse-engineered format (914 patients, Indian cohort).
3. "We got 0.82" — and why we refused to stop there.
4. The ablation design: one locked test set, one variable per experiment.
5. **The correction:** 62% was measurement (val vs test).
6. **The real drivers:** labels ≫ preprocessing ≫ 3D ≈ pretraining ≈ ensembling.
7. **The annotation gap:** two-sided cost, clinically validated — the star finding.
8. The L/R mirror-bug debugging vignette (crowd-pleaser).
9. We tested the masked-loss fix (E6): it beats the naive control and stabilises training but didn't beat the
   clean baseline here — the honest next test is a single-side evaluation.
10. Honest scope + the one transferable lesson: *interrogate your own metrics; the data is the product.*
