# S1 — SYNTHESIS: what actually drives OAR-segmentation performance on partial-label clinical data

**Every number below is on the SAME locked 43-case test set
(`Dataset002_Parotid/labelsTs`), scored against clean labels.**

Experiments: Track A = E1, E2, E2b, E5. Track B = P0, A1, E3, E4.
Sources: `ablation_study/<Eid>/RESULT.md`, master §20.A / §20.B.

---

## 0. THE HEADLINE, REWRITTEN

The project's famous result was **0.62 → 0.82 (+0.199 Dice)**, attributed loosely to "nnU-Net's 3D pipeline".
This study set out to disentangle that across four confounded axes. The answer:

> ## **62% of the 0.62 → 0.82 jump was never real. It was a measurement artifact.**
> The 0.62 was a **validation** number; the 0.82 a **test** number. Scored on the same locked test set, the
> Phase-1 models get **0.7434**, not 0.62. **The real improvement is +0.0753, not +0.199.**
>
> ## **And of what remains, the annotation gap — not 3D, not architecture — is the largest single driver.**

This is what P0 was for, and it is the study's most important result: **the headline the whole project was
built on was ~62% measurement.**

---

## 1. THE UNIFIED TABLE (every model, one test set, one column)

| # | Model / condition | Exp | Dice | Tversky | HD95 (mm) | Surf-Dice | Isolates |
|---|---|---|---|---|---|---|---|
| 1 | nnU-Net 5-fold ensemble | E5 | **0.8202** | 0.8174 | 5.24‡ | 0.8990 | ensembling |
| 2 | **nnU-Net 3d_fullres, 1 fold** | ref | **0.8187** | 0.8166 | 5.25‡ | 0.8965 | *the reference* |
| 3 | nnU-Net 2d, 1 fold | E1 | 0.8117 | 0.8085 | 5.52‡ | 0.8943 | dimensionality |
| 4 | nnU-Net 3d, dirty labels (430) | E2 | 0.7726 | 0.7644 | 5.45‡ | 0.8400 | labels (net) |
| 5 | Hand-built 3D U-Net + CC postproc | E4 | 0.7750 | 0.7788 | 6.05‡ | 0.8539 | architecture vs pipeline |
| 6 | Hand-built 3D U-Net (raw) | E4 | 0.7681 | 0.7746 | 26.76‡ | 0.8383 | (pre-postproc) |
| 7 | **Attention U-Net (scratch, 2D)** | P0 | **0.7434** | 0.7387 | 3.57* | 0.8827 | *Phase-1's real baseline* |
| 8 | U-Net (scratch, 2D) | P0 | 0.7390 | 0.7315 | 4.25* | 0.8702 | baseline |
| 9 | TransUNet (ImageNet-pretrained) | E3 | 0.7373 | 0.7325 | 3.78* | 0.8759 | pretraining |
| 10 | TransUNet (scratch, 2D) | P0 | 0.7313 | 0.7307 | 5.60* | 0.8594 | baseline |
| 11 | Swin-UNet (ImageNet-pretrained) | E3 | 0.7185 | 0.7122 | 6.70* | 0.8679 | pretraining |
| 12 | Swin-UNet (scratch, 2D) | P0 | 0.7156 | 0.7082 | 8.34* | 0.8794 | baseline |
| 13 | **nnU-Net 3d, GAPPED labels (208)** | E2b | **0.6899** | 0.6680 | 9.63‡ | 0.7649 | labels (latent) |

‡ anisotropic (0.977, 0.977, 3.0) · * isotropic 0.977 — **HD95 is not comparable between the two groups**
(master §12.3 / §14.5). Dice / Tversky / Surface-Dice are comparable throughout.

---

## 2. FINDING 1 — most of the headline was a val-vs-test artifact (P0)

The same four checkpoints, scored on Phase-1's validation set (master §13.2) vs the locked test set (P0):

| Model | VAL (§13.2) | TEST (P0) | Δ |
|---|---|---|---|
| U-Net | 0.6209 | 0.7390 | **+0.1181** |
| Attention U-Net | 0.6346 | 0.7434 | **+0.1088** |
| TransUNet | 0.6332 | 0.7313 | **+0.0981** |
| Swin-UNet | 0.5106 | 0.7156 | **+0.2050** |
| **mean** | **0.5998** | **0.7323** | **+0.1325** |

**The same weights score +0.13 higher on test than on validation.** Nothing about the models changed.

**Why — and this is the crux:** Phase-1's validation set was the 47 parotid-bearing val patients, which
**include single-side (partially annotated) cases**. The locked test set is 43 **both-parotid, QC-clean**
cases. Master §16.1 predicted exactly this: *"Partially-annotated cases penalise stronger models — a model
that correctly extends beyond the incomplete GT gets distance penalties for correct anatomy → reported Dice
is a conservative underestimate."*

**So the 0.62 was deflated by the annotation gap acting on the *evaluation* side.** The project's headline
compared a gap-deflated val number against a clean test number, and the difference was booked as progress.

**Consequence for the record:** *"0.62 → 0.82"* should never be quoted again without this caveat.
The defensible statement is **"0.7434 → 0.8187 on the same held-out test set (+0.075)"**.

---

## 3. FINDING 2 — decomposing the real +0.0753

Each row is a controlled A/B on the locked test set:

| Axis | Δ Dice | Share of +0.0753 | Experiment |
|---|---|---|---|
| **Label quality (net, as suffered)** | **+0.0461** | **~61%** | E2: dirty-430 0.7726 → clean 0.8187 |
| Preprocessing + nnU-Net training recipe | ~+0.0222 | ~29% | remainder (Phase-1 0.7434 → E2 0.7726, minus 3D) |
| Dimensionality (2D → 3D) | **+0.0070** | ~9% | E1: 2d 0.8117 → 3d 0.8187 |
| Ensembling (1 → 5 folds) | +0.0015 (**n.s.**, p=0.35) | ~2% | E5, paired Wilcoxon, worse on 17/43 |

**Labels dominate.** Not dimensionality (9%), not ensembling (2%, statistically zero).

**⚠️ This retracts the interim Track-A claim that preprocessing drove ~73%.** That figure was computed
against the **0.62 val** baseline before P0 existed. With Phase-1's true test score (0.7434), the picture
inverts: labels lead, preprocessing is second. Track A's §20.A entry also predicted *"the ordering of the
axes will not change"* once P0 landed — **that prediction was wrong; the ordering changed.** This is why the
val-vs-test correction was worth doing rather than assuming.

**Honesty about additivity:** these axes interact and the budget is approximate. E4 shows nnU-Net's *training
machinery* alone is worth **+0.0437** over a competent hand-built 3D U-Net on identical preprocessed data —
larger than the whole +0.0222 "preprocessing" remainder. So "preprocessing" here means *nnU-Net's entire
recipe* (resampling + CTNormalization + augmentation + deep supervision + LR schedule + patch sampling), not
resampling alone, and the split between "preprocessing" and "training machinery" is not cleanly separable
with the experiments run.

---

## 4. FINDING 3 — the annotation gap is the thread through the whole project

It is the only factor that shows up **everywhere**, on both sides of the ledger:

| Where it bites | Effect | Experiment |
|---|---|---|
| **Training** (latent, constant N) | **−0.1288 Dice**, HD95 5.25 → 9.63 | E2b |
| **Training** (net, with 2× data compensating) | −0.0461 Dice | E2 |
| **Evaluation** (deflates val scores) | **−0.1325 Dice** on the same weights | P0 vs §13.2 |
| Prevalence in the data | **222/430 = 51.6%** of parotid-bearing train/val patients are single-side | E2 scan |

**The gap costs ~0.13 Dice on the training side and ~0.13 on the evaluation side.** Both were previously
invisible; together they are larger than every architectural and pipeline choice in this study combined.

The three-way arm design pins down its structure:
- **clean − E2b = −0.1288** — the gap's *pure* cost at fixed N (the labels are the only variable)
- **E2 − E2b = +0.0827** — the 222 extra partially-labelled patients **buy back 64%** of the damage
- **clean − E2 = −0.0461** — the *net* cost in the real historical condition

**Both arms were necessary.** E2 alone would have reported −0.046 (understating it by half); E2b alone would
have reported −0.129 (overstating its historical role). The confound (0.083) was **larger than the effect E2
alone would have measured**.

**Actionable:** partially-labelled patients are **worth including** — they recover 64% of the damage and
restore boundary quality almost entirely (HD95 9.63 → 5.45 ≈ the 5.25 reference). But they still leave 4.6
points on the table versus clean labels. **`pipeline/masked_loss.py` (§10.2, written, unit-tested, never
used) is now the single best-motivated next experiment in the project**: masking un-annotated organs out of
the loss should capture all 430 patients' data benefit *without* the 12.9-point label penalty — plausibly
beating 0.8187. This is no longer a design intuition; it is a number-backed prediction.

---

## 5. FINDING 4 — architecture and initialisation barely matter

| Evidence | Result |
|---|---|
| **Six 2D models** (4 scratch + 2 pretrained), same data/protocol | span **0.0278 Dice** (0.7156–0.7434) |
| **ImageNet pretraining** (E3, TransUNet: 258/258 weights, identical 102.5M arch) | **+0.0060** Dice — noise |
| ImageNet pretraining (E3, Swin) | +0.0029 Dice — noise |
| Best *pretrained transformer* (0.7373) vs best *from-scratch CNN* (0.7434) | **CNN still wins** |
| **Hand-built plain 3D U-Net** on nnU-Net-preprocessed data (E4 + CC) | **0.7750** vs nnU-Net 0.8187 |
| 2D vs 3D (E1) | +0.0070 |
| 31M → 102M params (Phase-1, §13.4) | no gain |

**Q3 answered: no.** The transformers' Phase-1 weakness was *not* the from-scratch handicap. Give TransUNet
its intended ImageNet ResNet-50 — exact architecture, 258/258 weights — and it gains nothing measurable and
*still* loses to a 2018 Attention U-Net. This **half-refutes §13.4**: the data is indeed too little for a
ViT, but pretraining does not rescue it. Mechanism: the pretrained TransUNet drove **train loss to 0.0545**,
below the from-scratch model's best-ever *val* loss, while its own val sat at 0.1810 — it memorised the 6,934
parotid slices without generalising.

**Pretraining's one real benefit is boundaries:** HD95 −1.82 mm (TransUNet) and −1.64 mm (Swin), consistently
on both arms. Pretrained features localise gland *edges* better; they don't find more gland.

**Q4 answered (E4):** a plain hand-built 3D U-Net on nnU-Net's preprocessing reaches **0.7681**, and
**largest-connected-component postprocessing collapses HD95 from 26.76 → 6.05 mm** (≈ nnU-Net's 5.25) at
almost constant Dice. So **nnU-Net's boundary advantage is postprocessing, not architecture.** The residual
−0.0437 Dice is nnU-Net's *training machinery* (deep supervision, augmentation, schedule), not its network.

---

## 6. FINDING 5 — Swin-UNet was never as bad as it looked (A1 + P0)

Phase-1 recorded Swin as a catastrophic outlier: **val Dice 0.5106**, HD95 12.10 mm. Two experiments correct
this:

- **P0:** on the locked test set Swin scores **0.7156** — **+0.2050 over its val score, the biggest val→test
  jump of any model**, and only **0.028 behind the best CNN**. Its "collapse" was substantially a val-set
  artifact (the annotation gap again — Swin's bilateral predictions were punished hardest by partial GT).
- **A1:** its *boundary* failure is real and now explained. Both hypotheses confirmed, on 86 glands with true
  spacing:

| Metric | Swin | U-Net | paired p |
|---|---|---|---|
| mean connected components | **2.84** | 1.20 | 1.3e-11 |
| % multi-component | **75.6%** | 16.3% | — |
| mean stray islands | **1.85** | 0.23 | 1.1e-11 |
| farthest stray (mm) | **55.4** (max 306) | 12.2 | 1.9e-8 |
| HD95 full → largest-CC | 14.30 → **10.53** (−3.77) | 8.53 → 8.49 (−0.04) | — |

**H2 (stray islands) is dominant** — removing them drops Swin's HD95 by 3.77 mm vs 0.04 for U-Net.
**H1 (the 128→512 bilinear upsample workaround, §11.4) is the residual** — Swin-lcc at 10.53 still trails
U-Net's 8.49. **Q5 answered.**

---

## 7. THE FIVE QUESTIONS

| Q | Question | Answer |
|---|---|---|
| **Q1** | Is the 0.62 ceiling 2D vs 3D, or preprocessing/labels? | **Neither mainly — it was largely measurement.** Of the real +0.075, 3D is +0.007 (9%). The Phase-1 ceiling was *not* dimensionality. |
| **Q2** | How much is the annotation gap? | **The largest single driver.** −0.129 latent / −0.046 net on training, plus −0.133 on evaluation. 51.6% of parotid-bearing patients are affected. |
| **Q3** | Were the transformers handicapped by no pretraining? | **No.** +0.006 / +0.003 Dice with correct ImageNet init; both still lose to a from-scratch CNN. Boundaries improve (−1.8 mm), overlap doesn't. |
| **Q4** | Does architecture matter once the pipeline is right? | **Barely.** Six 2D models span 0.028. A hand-built 3D U-Net + trivial postproc matches nnU-Net's HD95; the −0.044 residual is training machinery, not architecture. |
| **Q5** | Why was Swin's HD95 so bad? | **Stray islands (dominant) + coarse 128→512 upsampling (residual).** Both confirmed at p < 1e-8. And its Dice "collapse" was mostly a val-set artifact. |

---

## 8. WHAT THIS MEANS

1. **Stop quoting 0.62 → 0.82.** Quote **0.7434 → 0.8187 (+0.075)** on the locked test set. The old figure
   is ~62% measurement artifact. This is the single most important correction the study produces, and it is
   the kind of thing that gets caught in review if you don't catch it yourself.
2. **The data is the product, not the model.** Labels (+0.046 net / +0.129 latent) and nnU-Net's recipe
   (+0.022) beat every architectural choice (0.028 across six models, +0.007 for 3D, +0.006 for pretraining,
   +0.0015 for ensembling). Effort spent on architectures was effort largely wasted; effort on annotation
   quality would pay ~20× more.
3. **Ship the single fold.** Ensembling is statistically indistinguishable from zero (p=0.35) at 5× the cost.
4. **Add connected-component postprocessing to anything hand-built.** It bought E4 a 20.7 mm HD95 improvement
   for free.
5. **Build `masked_loss.py` next.** It is the only intervention the evidence predicts could beat 0.8187.
   **⚠️ SUPERSEDED — DONE (see ADDENDUM — E6 below): masked loss was built and tested; it beats the naive
   control but did NOT beat 0.8187. And E7 tested the per-side "decompose" alternative (also < 0.8187) plus a
   held-out single-side eval. Net: no single-side-data strategy beats the clean both-parotid set.**
6. **The clinical finding is the contribution.** The annotation gap — Dr. Mod's confirmed observation that
   clinicians deliberately skip the healthy-side parotid — is not a footnote. It is the dominant quantitative
   factor in the entire study, on both the training and evaluation sides, and it is the part of this work
   that is genuinely novel relative to the published literature.

---

## 9. LIMITATIONS (state these; do not let a reviewer find them first)

1. **n = 43.** Per-case Dice std ≈ 0.078, so deltas below ~0.02 are noise. E1 (+0.007), E3 (+0.006/+0.003)
   and E5 (+0.0015) are all **within noise** — they are reported as "no measurable effect", not as zero.
   Only E2/E2b (−0.046 / −0.129), the val→test artifact (+0.133) and E4's residual (−0.044) clear it.
2. **Single fold, single seed** for E1/E2/E2b/E4/E3. No seed-variance estimate; run-to-run variation is
   unmeasured and could plausibly account for the sub-0.02 effects on its own.
3. **HD95 is not comparable across groups** — P0/E3/A1 use isotropic 0.977; nnU-Net/E2/E4 use true
   anisotropic (0.977, 0.977, 3.0). Never put them in one column. (Dice/Tversky/Surf-Dice are fine.)
4. **The attribution is not strictly additive.** Axes interact; "preprocessing" is a remainder that bundles
   nnU-Net's whole training recipe, and E4 shows the machinery alone exceeds it. Treat the shares as
   indicative ordering, not a precise budget.
5. **E2's arm has an N confound by construction** (208 → 430). That is why E2b exists; cite both.
6. **E3's Swin arm has an architecture confound** (timm Swin-T 48.3M, window 7/224 ≠ frozen Swin 27M,
   window 8/512). The **TransUNet arm is clean** and carries the pretraining claim.
7. **E3 budget deviations:** empty slices capped at 40/patient; 12,000 samples/epoch vs Phase-1's full pass.
   Affects absolute values; the pretrained-vs-scratch delta applies the same cap to both sides.
8. **The test set is the easy subset** — 43 both-parotid QC-clean cases. Real-world performance on
   single-side and messy cases is *not* measured by any number here. This cuts both ways: it is why the val
   scores looked bad, and it means 0.8187 is an optimistic operating point.
   **→ RESOLVED by E7 (2026-07-18):** the 58 held-out single-side test patients WERE scored (see ADDENDUM — E7).
   On the annotated at-risk gland the models score ~0.85 (nnU-Net 0.8523, experts 0.8552) — *higher* per-gland
   than the both-parotid average — so 0.8187 was not the optimistic ceiling this limitation feared. The
   single-side gap is now measured, not assumed.
9. **Phase-1 vs nnU-Net differ in training pool** (~383 parotid-bearing patients vs 208 both-annotated), so
   the "pipeline" delta carries a mild data-volume confound.

---

## 10. ONE-PARAGRAPH ABSTRACT

We ran a seven-experiment ablation to explain a reported 0.62 → 0.82 Dice improvement in parotid
auto-segmentation on 914 head-and-neck CTs from an Indian hospital cohort. **Re-scoring the original models
on the held-out test set shows ~62% of that improvement was a validation-vs-test measurement artifact: the
true gain is 0.743 → 0.819 (+0.075).** Decomposing the remainder against a locked 43-case test set, the
dominant factor is **label quality** (+0.046 net; +0.129 when isolated at constant training-set size),
followed by preprocessing/training recipe (+0.022), dimensionality (+0.007), and ensembling (+0.0015, not
significant, p=0.35). Architecture and initialisation are near-irrelevant: six 2D models span 0.028 Dice,
ImageNet pretraining adds +0.006 (though it improves HD95 by 1.8 mm), and a hand-built 3D U-Net matches
nnU-Net's boundary accuracy once trivial connected-component postprocessing is applied. The single largest
effect in the study is **the annotation gap** — clinicians deliberately leave the healthy-side parotid
un-contoured in 51.6% of cases — which costs ~0.13 Dice in training *and* deflates validation scores by
~0.13, and which a masked partial-label loss should be able to recover.

---

*Mirrored in master §20.C. Per-experiment detail: `ablation_study/<Eid>/RESULT.md`.*

---

## ADDENDUM — E6: masked partial-label loss (2026-07-17)

§4 and §8 named `masked_loss.py` "the single best-motivated next experiment... plausibly beating 0.8187." E6
ran it. **The mechanism holds in direction; the strong prediction does not.** Three arms on the E4 custom
3D U-Net, same locked 43-case test set, +CC postproc (2× L40S, ~$2):

| Arm | Train | Loss | Dice +CC |
|---|---|---|---|
| clean-208 (=E4) | 208 both | Dice+BCE | **0.7750** |
| **masked-430** | 430 | Masked | **0.7589** |
| dirty-430 | 430 | Dice+BCE | 0.7296 |
| _ref:_ nnU-Net 1 fold | 208 | — | 0.8187 |

- **masked-430 > dirty-430 (+0.029):** masking the un-annotated gland beats penalising it on the same 430
  patients — the annotation-gap fix works *in the expected direction*.
- **masked-430 (0.7589) < clean-208 (0.7750) < nnU-Net (0.8187):** it did **not** beat the clean 208-case
  baseline or nnU-Net. The prediction that masking would land above 0.8187 is **not confirmed** here.
- Both deltas are within/near the ≈0.05 paired-noise floor (n=43, single fold/seed); the dirty control
  **diverged** mid-training (best ckpt @ ep45), confounding masked-vs-dirty with stability; and — decisively —
  **the test set is all both-parotid clean**, so it structurally cannot reward the masked loss's main expected
  benefit, single-side cases (limitation 8). A secondary observation worth keeping: the masked arm trained
  *stably* while the dirty arm diverged, plausibly because the ordinary loss's conflicting gradients (punishing
  correct anatomy) destabilise training.

**Revised "best next step":** masking is a *directionally correct, stability-improving* fix, not a demonstrated
0.82-beater. To actually test §4's claim, the missing instrument is a **single-side / mixed test set** (or
per-side evaluation on single-side patients) where the gap's evaluation-side cost is live — that is where
masking should visibly pay off. Retraining the dirty control with gradient clipping + multiple seeds would
also de-confound the +0.029. Full record: `ablation_study/E6_masked_loss/RESULT.md`.

---

## ADDENDUM — E7: single-side evaluation + per-side specialist experts (2026-07-18)

Fills the study's biggest hole (limitation 8 / §9.8): every prior number was on the 43 both-parotid cases.
E7 scored the **58 held-out single-side TEST patients** (30 only_R + 28 only_L — verified from raw data, L/R
naming validated 58/58 against geometry, zero training leakage) and added a 4th annotation-gap treatment.

**Part 1 — existing models on the single-side 58 (annotated-side Dice+CC):** nnU-Net clean-208 **0.8523**,
E4 0.8011, dirty-430 0.7436, masked-430 0.8010. **Headline: the at-risk annotated gland is segmented WELL
(~0.85), *higher* per-gland than the both-parotid average** — the realistic single-side cases are not a
failure mode; if anything the annotated gland is the easier target (bigger/clearer). The nnU-Net arms beat the
custom U-Net arms, same architecture ordering as the both-parotid set.

**Contralateral-prediction rate (the §16 suppression figure, on raw preds):** dirty-430 predicts the healthy
un-annotated gland least (87.9 %), masked-430 recovers it (94.8 %), clean/expert arms 96–100 %. §16's
annotation-gap suppression is **confirmed directionally but modest** — every model predicts the contralateral
gland in ≥88 % of cases, so the gap does not strongly suppress healthy anatomy at inference.

**Part 2 — per-side specialist experts ("decompose", NOT a mixture-of-experts):** two single-class nnU-Net
`3d_fullres` experts (left 320, right 318 patients), trained 2× A40 parallel, combined by hemifield tie-break.
- Both-parotid 43: **0.8099** — competitive but does not beat clean-208's 0.8187. Completes the four-way gap
  treatment: **discard 0.8187 · penalise (dirty) · mask 0.7589 · decompose 0.8099.** No strategy beats simply
  discarding the single-side data on the easy set — a clean, if humbling, result.
- Single-side 58: **0.8552 — the best of all arms**, narrowly over clean-208 (0.8523). The experts' genuine
  win is exactly here (realistic cases + design cleanliness), not the both-parotid set.
- **Combining is lossless** (combined per-side = each standalone expert; no cross-side interference), validating
  the decomposition.

**Verdict:** the annotation gap's *evaluation-side* cost that E6 predicted should appear on single-side data is
**smaller than expected** — models handle the annotated side well and mostly predict the healthy gland anyway.
Experts are the honest incremental win on the previously-unmeasured single-side cases (a data-strategy result,
320/318 vs 208 patients — not a like-for-like architecture win; do not oversell). Full record:
`E7_singleside/RESULT.md`, `E7_experts/RESULT.md`. Cost ~$8 (2× A40; training ≈$3.4, rest = upload).
