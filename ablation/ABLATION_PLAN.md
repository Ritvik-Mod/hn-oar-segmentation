# ABLATION STUDY — PLAN & RATIONALE (what to run now, and why)

> **Read this together with `MASTER_PROJECT_REFERENCE.md`** (the full project brain, at the repo root).
> That document is the source of truth for the dataset, the existing code, and all prior results. This
> document defines the *new* controlled experiments to run and the questions they answer. Two Claude Code
> agents will execute it in parallel — see `TRACK_A_nnUNet_tasks.md` and `TRACK_B_custom_tasks.md`.
> How the agents log progress is governed by `AGENT_INSTRUCTIONS.md`.
>
> Owner: Ritvik Mod. Compiled 2026-07-16.

---

## 0. WHY THIS STUDY EXISTS (the questions I want answered)

The headline result so far is **0.62 (Phase-1, four from-scratch 2D models) → 0.82 (Phase-2, nnU-Net 3D
ensemble)**. That jump is real but **scientifically unsatisfying as stated**, because it bundles *four*
changes at once and I cannot currently attribute the improvement to any single cause. I want to disentangle
them — for my own understanding first, and to make the conference presentation a rigorous *study of what
drives OAR-segmentation performance on partial-label clinical data*, not a horse-race.

**The confound.** The nnU-Net result differs from the from-scratch baselines on FOUR axes simultaneously:
1. **Dimensionality** — 2D slice-level vs full 3D.
2. **Preprocessing** — naive (raw 2D slices, single HU window, no resampling) vs nnU-Net's resampling to a
   common voxel spacing + CTNormalization + strong augmentation.
3. **Label quality** — the baselines trained on *all* H&N patients including cases where a present parotid
   was **not contoured** (treated as background = the model punished for correct predictions); nnU-Net
   trained only on the **both-parotid-annotated, QC-cleaned** subset (Dataset002).
4. **Ensembling** — a single model vs a 5-fold ensemble.

**The specific open questions:**
- **Q1.** Is the ~0.62 ceiling caused by 2D vs 3D, or by preprocessing/labels? (i.e. how much does *3D*
  alone actually buy?)
- **Q2.** How much of the gain is the **annotation gap** (dirty partial labels) — the problem I identified
  that was *present during baseline training*? This is my most original question.
- **Q3.** Did training the two transformers (**TransUNet, Swin-UNet**) **from random init with no ImageNet
  pretraining** unfairly handicap them? Is their weakness architectural or just "from scratch"?
- **Q4.** Does architecture matter at all once the pipeline is fixed, or is nnU-Net's win almost entirely
  pipeline/data/ensembling? (nnU-Net's own `3d_fullres` is a *plain* ~30M conv U-Net, not exotic.)
- **Q5 (analysis, no training).** *Why* did Swin-UNet do disproportionately badly on **HD95 (12.1 mm)** and
  Surface-Dice (0.63) relative to its Dice (0.51)?

---

## 1. GROUND RULES (apply to EVERY experiment)

1. **The test set stays locked and identical everywhere.** All models are finally scored on the **same
   held-out test set** = the 43 both-parotid, QC-clean cases in `Dataset002_Parotid/labelsTs`, using
   `pipeline/eval_testset.py` (3D Dice / clinical Tversky / HD95 / Surface-Dice, true anisotropic spacing).
   **Never** train on the test patients. **Never** change the test set between experiments (only the
   *training* condition changes) — otherwise the numbers aren't comparable.
2. **One variable per experiment.** Each run isolates a single axis; hold everything else fixed.
3. **Everything is compared on TEST, not VAL.** This means the four existing Phase-1 checkpoints must be
   **re-evaluated on the locked test set** (they were only ever scored on val) so the whole study shares one
   comparable column. (Task in Track B.)
4. **Do not touch** the existing checkpoints, `ML_Dataset_Final`, the split JSONs, the `dataset.h5`, or the
   web-app/showcase folders. Read-only. New outputs go under `ablation_study/<Eid>/`.
5. **nnU-Net always uses the no-mirror custom trainer** (`nnUNetTrainer_250epochs_noMirror`) and
   `--disable_tta` at inference. Re-enabling mirroring reintroduces the L/R bug (see master doc §15).
6. **Reproducibility:** record seed, git/commit or file versions, exact command, GPU, and wall-time in each
   experiment's `RESULT.md`.

---

## 2. THE EXPERIMENT MATRIX

| ID | Question | What changes | Held fixed | Baseline to compare against | Track |
|----|----------|--------------|------------|-----------------------------|-------|
| **E1** | Q1: 2D vs 3D | nnU-Net `2d` config | Dataset002, preprocessing, single fold, noMirror | nnU-Net `3d_fullres` single fold (0.8187) | **A** |
| **E2** | Q2: annotation gap | Training data = "dirty" full-label set (single-side parotids, omitted side left as background) | nnU-Net `3d_fullres`, single fold, **same clean test set** | Dataset002 clean `3d_fullres` single fold (0.8187) | **A** |
| **E3** | Q3: transformer pretraining | TransUNet + Swin-UNet **with ImageNet-pretrained backbones** | Same 2D data/protocol as Phase 1 | The from-scratch TransUNet/Swin **re-scored on test** | **B** |
| **E4** | Q4: architecture vs pipeline | A **hand-built 3D U-Net** trained on nnU-Net-preprocessed data | Dataset002, 3D, single fold | nnU-Net `3d_fullres` single fold (0.8187) | **B** |
| **E5** | (bonus) ensembling | single fold vs 5-fold | Dataset002, `3d_fullres`, noMirror | already have both (0.8187 vs 0.8202) | **A** |
| **A1** | Q5: Swin boundary failure | *analysis only, no training* | — | existing Swin predictions/checkpoint | **B** |
| **P0** | make study comparable | re-evaluate the 4 existing Phase-1 checkpoints on the **test** set | — | their published val numbers | **B** |
| **S1** | synthesis | assemble the unified comparison table + write the reasoning | needs A **and** B done | — | either (last) |

Only **S1** depends on both tracks; it runs at the end and blocks nothing.

---

## 3. PER-EXPERIMENT DETAIL

### E1 — Dimensionality: 2D vs 3D  *(Track A, nnU-Net)*
- **Method:** nnU-Net already planned Dataset002. Train the **`2d`** configuration with the no-mirror
  trainer, single fold 0, `--npz`. Predict the test set (`--disable_tta`), evaluate with `eval_testset.py`.
- **Compare:** E1 (2d) vs the existing `3d_fullres` single-fold (0.8187) — *everything else identical*.
- **Expected:** if `2d` lands ≥ ~0.75, then 3D buys only a few points and the ceiling was **preprocessing +
  labels**, not dimensionality. If `2d` collapses toward 0.62, then 3D context is the main driver.
- **Proves:** the true value of the 2D→3D axis, cleanly, with zero custom code.
- **Cost:** low (2d is fast; ~1–2 h on a 4090).

### E2 — Annotation gap: clean vs dirty labels  *(Track A, nnU-Net)* — **the original contribution**
- **Method:** build a new **"dirty" training dataset (Dataset003)** = all TRAIN/VAL patients that have *at
  least one* parotid, using their labels **as-is** — i.e. single-side patients keep only the annotated side,
  and the un-contoured contralateral gland is left as **background** (this deliberately reproduces the
  annotation-gap noise the baselines suffered). Reuse `pipeline/build_volumes.py` + a modified
  `make_nnunet_dataset.py` (drop the "both-annotated only" filter; keep the QC volume filter). **Keep the
  TEST set exactly the clean 43-case Dataset002 test set** — do not change it. Train `3d_fullres`, single
  fold, noMirror; evaluate on the clean test set.
- **Compare:** E2 (dirty-trained) vs Dataset002 clean-trained `3d_fullres` single fold (0.8187).
- **Expected:** dirty-trained model scores meaningfully lower and/or shows more one-sided misses → quantifies
  the cost of the annotation gap.
- **Proves:** how much the partial-label noise alone degrades performance — directly answers Q2 and
  motivates the `masked_loss.py` design. *(Optional stretch: also train with `masked_loss.py` on the same
  dirty set to show the fix recovers the gap — only if time permits; it needs custom training, so if done,
  hand that variant to Track B.)*
- **Cost:** medium (dataset build is CPU/minutes; one `3d_fullres` fold ~1–3 h).

### E3 — Transformer pretraining  *(Track B, custom PyTorch)*
- **Method:** retrain **TransUNet** with an **ImageNet-pretrained ResNet-50 + ViT** backbone and **Swin-UNet**
  with **ImageNet-pretrained Swin-T** weights — their *intended* configuration. Keep the Phase-1 protocol
  identical otherwise (combined Dice+BCE, Adam 1e-4, batch 8, AMP, same 2D dataloader/weighting). Note:
  input is single-channel CT 512×512 — adapt the pretrained stem (repeat/average the 3-channel weights to
  1-channel, or duplicate the channel). Evaluate **on the test set** with the Phase-1 3D volumetric eval.
- **Compare:** pretrained vs the from-scratch TransUNet/Swin, **both scored on the test set** (see P0).
- **Expected:** pretraining lifts both transformers, especially boundary metrics; Swin should improve most.
  If they still don't beat the CNNs, that itself is a finding (data-bound ceiling dominates).
- **Proves:** whether the transformers' weakness was "from scratch" (Q3) or genuinely the architecture.
- **Cost:** medium (2 × 2D transformer trainings; free on HPC when a GPU is free, else ~$3–6 cloud).

### E4 — Architecture vs pipeline: hand-built 3D U-Net  *(Track B, custom PyTorch)*
- **Method:** implement a **plain 3D U-Net** (extend the existing `unet.py` to 3D conv/pool, or a compact
  3D encoder-decoder) and train it on **nnU-Net-style preprocessed 3D volumes** of Dataset002 (resampled to
  the same [3.0, 0.977, 0.977] spacing, CTNormalization, patch-based training). Single fold; evaluate on the
  test set with `eval_testset.py`.
- **Compare:** custom 3D U-Net vs nnU-Net `3d_fullres` single fold (0.8187) — *same data, same
  preprocessing, different implementation*.
- **Expected:** it lands close to nnU-Net (~0.78–0.82). If so → **architecture is nearly irrelevant once the
  pipeline is right; nnU-Net's win is preprocessing + 3D + ensembling.** That is the strongest thesis of the
  study.
- **Proves:** Q4. *(If replicating nnU-Net preprocessing is too costly, a cheap proxy is nnU-Net's own
  residual-encoder preset `-p nnUNetResEncUNetLPlans` as a "different, stronger architecture, same pipeline"
  run — note this is a Track-A nnU-Net task if used. Prefer the true custom 3D U-Net for the cleaner claim;
  fall back to the preset only if blocked.)*
- **Cost:** medium–high (custom 3D training; the most involved run).

### E5 — Ensembling  *(Track A, mostly analysis)*
- Single fold (0.8187) vs 5-fold ensemble (0.8202) are **already computed**. Just tabulate the delta and note
  it (small here). If any fold is missing, that's already covered by the main project's 5-fold results.

### A1 — Why Swin-UNet's boundary metrics are bad  *(Track B, analysis only, NO training)*
- **Hypotheses to confirm from existing artifacts (master doc §11.4, §13):**
  1. Swin's final prediction is made at **128×128 then bilinearly upsampled to 512×512** (the memory
     workaround replacing `FinalPatchExpanding`) → inherently coarse boundaries → inflated HD95 / low
     Surface-Dice even when overlap is okay.
  2. Swin **underfits** (val loss floored at 0.21 vs ~0.16 for the CNNs) → scattered false-positive voxels
     far from the gland → HD95 (a 95th-percentile distance) blows up.
- **Method:** on the existing Swin predictions/checkpoint, (a) count connected components per predicted
  gland (expect many stray islands), (b) measure distance of stray components from the main gland, (c) show
  a montage of a coarse/island-y Swin prediction vs a CNN prediction. No GPU/training needed.
- **Proves:** a concrete, code-grounded explanation for Q5 to put in the write-up.

### P0 — Re-evaluate the 4 Phase-1 checkpoints on the TEST set  *(Track B)*
- Run each existing `.pth` (U-Net, Attention, TransUNet, Swin) through the test set with the Phase-1 3D
  volumetric eval so the whole study reports **test** numbers side-by-side (fixes the current val-vs-test
  apples-to-oranges). Cheap, CPU-ok. This also lets the conference abstract quote Phase-1 on test.

### S1 — Synthesis  *(run last, after A and B finish)*
- Assemble one master comparison table (all models, all on the test set) with the axis each row isolates.
- Write the reasoning: attribute the 0.62→0.82 gap across {3D, preprocessing, labels, ensembling,
  architecture, pretraining}. Update the master doc §20 and the conference materials if desired.

---

## 4. OUTPUT LAYOUT (every experiment)

```
ablation_study/
  E1_2d_vs_3d/            RESULT.md  +  eval CSV  +  (model dir or pointer)  +  logs
  E2_annotation_gap/      RESULT.md  +  eval CSV  +  dataset-build notes  +  logs
  E3_transformer_pretrain/RESULT.md  +  eval CSV  +  checkpoints  +  logs
  E4_custom_3d_unet/      RESULT.md  +  eval CSV  +  checkpoint  +  logs
  A1_swin_failure/        RESULT.md  +  figures/tables
  P0_phase1_on_test/      RESULT.md  +  eval CSV
  SYNTHESIS.md            (the unified table + reasoning)
```
Each `RESULT.md`: the question, exact command, config, seed, GPU, wall-time, the metric table, and a 2–4
sentence interpretation.

---

## 5. LIVE STATUS (agents update their own rows only — see AGENT_INSTRUCTIONS.md)

| ID | Track | Status | Test Dice | HD95 | Notes | Updated |
|----|-------|--------|-----------|------|-------|---------|
| E1 | A | ✅ done | **0.8117** | **5.52** | **2D ≈ 3D: only −0.0070 Dice vs the 0.8187 3d_fullres baseline** (Surf-Dice 0.8943 vs 0.8965 — effectively identical). Pre-registered guess was 0.72–0.78; it beat that. **Dimensionality accounts for ~0.7 of the ~20-point 0.62→0.82 gain (~3.5%)** — the Phase-1 ceiling was NOT 2D. L40S, 26.4 s/epoch × 250 ≈ 110 min. Internal val pseudo-Dice [0.833, 0.849] (no mirror bug). See `E1_2d_vs_3d/RESULT.md`. | 2026-07-16 |
| E2 | A | ✅ done (both arms) | **E2 0.7726 / E2b 0.6899** | **5.45 / 9.63** | **The gap's LATENT cost is −0.1288 Dice (E2b, constant-N: clean 0.8187 → 0.6899, HD95 5.25→9.63), but the 222 extra single-side patients buy back +0.0827 (64%), leaving a NET −0.0461 (E2 dirty-430 = 0.7726, HD95 5.45 ≈ baseline).** Both arms were necessary: E2 alone understates the gap (−0.046), E2b alone overstates its historical role (−0.129); the confound (0.083) was **larger than the effect E2 would have reported**. ⚠️ **Correction:** an earlier entry claimed the gap explains ~65% of the 0.62→0.82 gain — **retracted**; Phase-1 trained on ~383 parotid patients, so its label condition matches **E2**, not E2b. **Corrected attribution: preprocessing ≈ +0.145 (~73%) ≫ labels ≈ +0.046 (~23%) ≫ 3D +0.007 > ensembling +0.0015 (n.s.).** Preprocessing was the driver — proving §13.4's suspicion. Tversky falls further than Dice in both arms (false-negative signature); R/L asymmetry widens monotonically with label damage (clean 0.018 → E1 0.031 → E2 0.033 → E2b 0.044). **⇒ strongest case yet for `masked_loss.py`**: it should capture all 430 patients' data benefit *without* the 12.9-pt penalty. See `E2_annotation_gap/RESULT.md`. | 2026-07-16 |
| E5 | A | ✅ done | 0.8202 vs 0.8187 | 5.24 vs 5.25 | Ensemble delta +0.0015 Dice is **not significant** (paired Wilcoxon p=0.35, 26/17 win/loss, n=43); HD95 null too (p=0.26). Ensembling explains ~none of the 0.62→0.82 gain. | 2026-07-16 |
| E3 | B | ✅ done | TransUNet **0.7373** / Swin **0.7185** | 3.78 / 6.70 | **ANSWER TO Q3: NO — pretraining does not fix the transformers.** TransUNet 0.7313→**0.7373 (+0.0060)**, Swin 0.7156→**0.7185 (+0.0029)** — both within per-case noise (std≈0.078). **But boundaries clearly improve: HD95 −1.82 mm / −1.64 mm on both arms.** Ranking unchanged — best pretrained transformer (0.7373) **still loses to from-scratch Attention U-Net (0.7434)**. ⇒ the transformers' Phase-1 weakness was **not** the from-scratch handicap; the **data** is the ceiling (half-refutes §13.4). **All six 2D models now span just 0.0278 Dice (arch + init combined)** vs preprocessing ≈+0.145 and labels ≈+0.129. Pre-registered "Swin most" ❌ (TransUNet gained more on test); "boundaries improve" ✅ (strongest effect); "still don't beat the CNNs" ✅. ⚠️ **Val loss and test Dice disagree on BOTH arms** (TransUNet val 0.1692→0.1810 worse yet test better; Swin val 0.2101→0.1703 much better yet test flat) — capped-h5 batch composition + 2D-val vs 3D-test; **only test Dice is authoritative**. TransUNet train loss 0.0545 ≪ val 0.1810 = memorised, didn't generalise. ⚠️ Swin arm carries a documented arch confound (timm Swin-T 48.3M ≠ frozen Swin 27M); TransUNet arm is clean (258/258, identical 102.5M). Early stop @23 / @22; 94.3 / 55.5 min on 1× L40S; ~$2.7. See `E3_transformer_pretrain/RESULT.md`. | 2026-07-17 |
| E4 | B | ✅ done | 0.7681 raw / **0.7750 +CC** (vs 0.8187) | 26.76 raw / **6.05 +CC** (vs 5.25) | Plain 3D U-Net (16.5M), nnU-Net preproc (CTNorm from plans.json, no resample), fold-0 166/42, early-stop ep83, ~50min. Plain arch on right pipeline → 0.77, recovering most of the 2D(0.74)→nnU-Net(0.82) gap. **largest-CC postproc collapses HD95 26.76→6.05 (≈nnU-Net 5.25) at ~constant Dice** → boundary gap = postproc not architecture; residual −0.044 Dice = nnU-Net training machinery. Comparable via `eval_testset.py` (aniso). | 2026-07-17 |
| A1 | B | ✅ done | n/a | n/a | **Both hypotheses confirmed.** Swin vs U-Net (86 glands): 2.84 vs 1.20 components, 75.6% vs 16.3% multi-component, stray islands 55.4 vs 12.2 mm away (all p<1e-8). HD95 full→largest-CC drops **3.77mm for Swin vs 0.04 for U-Net** (H2 stray islands = dominant driver); residual Swin-lcc 10.53 vs U-Net 8.49 = H1 coarse 128→512 upsampling. Montages saved. | 2026-07-17 |
| P0 | B | ✅ done | Attn 0.7434 / U-Net 0.7390 / TransUNet 0.7313 / Swin 0.7156 | 3.57 / 4.25 / 5.60 / 8.34 (iso) | All 4 on locked 43-case test (n=86 sides). Cluster **0.72–0.74** (>0.62 val, <0.82 nnU-Net) → architecture spread <0.03 Dice. Swin worst HD95 despite mid Dice + fewest one-sided misses → pure boundary failure (→A1). TransUNet ckpt was truncated in upload, re-uploaded+rerun. Dice comparable to 0.8187; HD95 iso caveat. | 2026-07-17 |
| S1 | — | ✅ done | n/a | n/a | **`SYNTHESIS.md` + master §20.C written.** ⚠️ **HEADLINE RETIRED: ~62% of "0.62→0.82" was a val-vs-test measurement artifact** — P0 shows the same Phase-1 weights score **+0.1325 higher on test than val** (mean 0.5998→0.7323), because Phase-1's val set contained single-side partial-label cases while the locked test set is both-parotid QC-clean (§16.1 predicted exactly this). **Defensible claim: 0.7434 → 0.8187 (+0.0753).** Decomposition of the real gain: **labels +0.0461 (~61%)** ≫ preprocessing/recipe ~+0.0222 (~29%) ≫ 3D +0.0070 (~9%) > ensembling +0.0015 (n.s., ~2%). **Retracts** Track A's interim "preprocessing ~73%" claim (it used the 0.62 val baseline) and its prediction that the ordering wouldn't change — **it inverted**. The annotation gap is the thread: −0.129 latent / −0.046 net on training **plus −0.133 on evaluation**, affecting 51.6% of patients ⇒ `masked_loss.py` is the best-motivated next experiment. Architecture+init span just 0.028 across six 2D models. ⚠️ n=43, per-case std≈0.078 → deltas <0.02 (E1, E3, E5) are within noise. | 2026-07-17 |

Legend: ⬜ not started · 🟡 in progress · ✅ done · ⚠️ blocked/issue.
