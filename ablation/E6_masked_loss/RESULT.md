# E6 — Masked partial-label loss: does masking the un-annotated gland recover the annotation-gap cost?

**Question.** The ablation's star finding is the **annotation gap**: 51.6% of parotid-bearing
patients have only one gland contoured, and an ordinary loss penalises the model for correctly
predicting the un-contoured side (E2b latent −0.129 / E2 net −0.046 Dice). A **masked partial-label
loss** (`pipeline/masked_loss.py`) computes the loss only over the glands actually annotated per
patient, so it should use **all 430 patients' data** *without* the label penalty — §21's number-backed
prediction that this is the one intervention that could beat the clean reference (0.8187).

**Design — three arms on the SAME E4 hand-built 3D U-Net, one locked 43-case test set:**

| Arm | Train data | Loss | Status |
|-----|-----------|------|--------|
| **clean-208** | Dataset002 both-annotated (208) | ordinary Dice+BCE | ✅ EXISTS = E4 (0.7681 raw / 0.7750 +CC) — not retrained |
| **dirty-430** | all 430 parotid-bearing, single side left as background | ordinary Dice+BCE | ⬜ TRAIN (control) |
| **masked-430** | all 430 | `MaskedPartialLoss` (mask un-annotated channel) | ⬜ TRAIN — the key result |

Everything but **train scope** and **loss** is held identical to E4: architecture, preprocessing,
patch [48,224,192], Adam 1e-4, AMP, batch 2, 250 epochs/250 iters, ReduceLROnPlateau, seed 42,
nnU-Net fold-0 split, and the same `pipeline/eval_testset.py` scorer (true anisotropic spacing).

**Claim it proves:** if `masked-430 > dirty-430` and `masked-430 ≥ clean-208`, masking captures the
222 extra patients' data benefit without the annotation-gap penalty.

---

## Status: ✅ DONE (2026-07-17, RunPod 2× L40S, parallel) — mechanism partially confirmed

**Verdict in one line:** masked-430 (+CC **0.7589**) **beats** the dirty-430 control (**0.7296**, +0.029) —
masking the un-annotated gland helps versus penalising it — **but does NOT reach clean-208 (0.7750)** or
nnU-Net (0.8187). §21's strong prediction ("should beat 0.8187") is **not** confirmed on this test set. The
mechanism direction holds; the headline win does not. Both deltas are within/near the n=43 noise floor
(≈0.05), and the dirty control diverged mid-training (see caveats), so treat as suggestive.

Cost: 2× L40S in one pod, both arms in parallel, ~52 min training each + ~10 min eval ≈ **~$2** of the
$13.91 balance. Seed 42, single fold, nnU-Net fold-0 (344 train / 86 val over the 430).

Everything that can be done for **$0 on CPU** is done and verified. Both GPU runs are staged and
ready; the owner is deciding HPC-vs-RunPod (task §5). Nothing has been trained; no budget spent.

### What is built (all under `ablation_study/E6_masked_loss/`, copied from frozen E4 originals)
- `unet3d.py`, `loss3d.py`, `postproc_largestcc.py` — **verbatim copies of E4** (identical model + ordinary loss + CC postproc).
- `preprocess_e6.py` — E4's exact preprocessing, but training source = **Dataset003_ParotidDirty (430)**
  and test = clean **Dataset002 test (43)**. Reuses E4's `norm_constants.json` **verbatim** (clip [8087,8394],
  (x−8198.65)/57.49) so all three arms are normalised identically. **Adds a per-CASE annotation mask**
  `[mask_r, mask_l]` derived from each patient's **full-volume** labelmap (not per-patch) and stored in the npz.
- `dataloader_e6.py` — E4's patch sampler + oversampling + fold-0 split, extended to emit `(img, target, mask)`.
  On the L/R hflip augmentation the **mask is swapped together with the R/L target channels** (an only-R patient
  mirrored becomes only-L — its annotated gland moves to the L channel, so the mask must flip too).
- `train_e6.py` — one script, two arms via `--loss {dirty,masked}`. `dirty` uses `DiceBCELoss3D` (ignores mask,
  penalises the un-annotated side as background → reproduces the gap on 430); `masked` uses
  `pipeline/masked_loss.MaskedPartialLoss` (ignores the un-annotated channel).
- `predict_e6.py` — E4's sliding-window inference (Gaussian-weighted, 50% overlap), writes labelmaps in
  labelsTs geometry for `eval_testset.py`. Prediction is loss-agnostic → used for both arms.
- `sanity_mask.py` — the end-to-end mask-plumbing gate.

### What is verified ✅ (local, CPU)
1. **Environment:** torch 2.12.1, MPS available (Mac); pod will use CUDA/AMP as E4 did.
2. **Dataset003 geometry** matches E4's no-resampling assumption exactly: affine diag (0.977, 0.977, 3.0),
   labels {0,1,2}.
3. **`masked_loss.py` unit test** passes (corrupting an unannotated channel leaves the loss identical;
   all-annotated mask == plain combined loss).
4. **Preprocessing** produced **430 train + 43 test** npz (**2.0 GB** compact cache), norm mean −1.775 (≈ E4's
   −1.78). Annotation status: **both=208, only_R=110, only_L=112 → single-side 222/430 = 51.6%** (exact match to
   the study). Audit in `preprocessed/case_masks.json`.
5. **`sanity_mask.py` — ALL CHECKS PASS end-to-end through the dataloader:**
   - [A] per-case masks match annotation status; counts 208/110/112, zero "none".
   - [B] hflip swaps the mask with the R/L channels (`seen_flip=True`); the un-annotated channel is *always*
     background in every drawn patch.
   - [C] on a single-side sample, corrupting the un-annotated channel leaves **MaskedPartialLoss unchanged
     (Δ=0.0e0)** but shifts ordinary DiceBCE by 0.167 — the mask genuinely removes that gland from the gradient.
   - [D] on a both-annotated sample, masked == ordinary loss (Δ=1.6e−6) — masking is a no-op when fully labelled.

### Compute / upload plan (per task §5 + POD_UPLOAD_PLAYBOOK.md)
- Upload only the **2.0 GB `preprocessed/` cache** (not 13 GB raw NIfTI). masked-430 first, then dirty-430.
- Cheapest adequate GPU, **single fold**. Expected ~2–3 GPU-h/arm (E4 was ~50 min on an L40S).
- clean-208 already exists (free) — do not retrain.

---

## EXACT COMMANDS (ready to run on the GPU box)

```bash
# 0. (once) preprocess locally — ALREADY DONE on the Mac; upload preprocessed/ instead.
python3 preprocess_e6.py --out ./preprocessed          # 430 train + 43 test, ~2 GB, CPU

# 1. KEY ARM — masked-430 (run first)
python3 train_e6.py --loss masked --preproc ./preprocessed/train --save-dir ./ckpt_masked430
python3 predict_e6.py --ckpt ./ckpt_masked430/best_model.pth --preproc ./preprocessed/test --out ./pred_masked430
python3 ../../pipeline/eval_testset.py --pred-dir ./pred_masked430 \
    --gt-dir "../../Parotid-Project/Datasets/Dataset002_Parotid/labelsTs" \
    --model-name Custom_3D_UNet_masked430 --results-csv ./eval.csv
# +CC postproc (report raw AND +CC, like E4)
python3 postproc_largestcc.py --pred-dir ./pred_masked430 --out ./pred_masked430_cc
python3 ../../pipeline/eval_testset.py --pred-dir ./pred_masked430_cc \
    --gt-dir "../../Parotid-Project/Datasets/Dataset002_Parotid/labelsTs" \
    --model-name Custom_3D_UNet_masked430_CC --results-csv ./eval.csv

# 2. CONTROL ARM — dirty-430 (run only if budget allows, task §5)
python3 train_e6.py --loss dirty --preproc ./preprocessed/train --save-dir ./ckpt_dirty430
python3 predict_e6.py --ckpt ./ckpt_dirty430/best_model.pth --preproc ./preprocessed/test --out ./pred_dirty430
python3 ../../pipeline/eval_testset.py --pred-dir ./pred_dirty430 \
    --gt-dir "../../Parotid-Project/Datasets/Dataset002_Parotid/labelsTs" \
    --model-name Custom_3D_UNet_dirty430 --results-csv ./eval.csv
python3 postproc_largestcc.py --pred-dir ./pred_dirty430 --out ./pred_dirty430_cc
python3 ../../pipeline/eval_testset.py --pred-dir ./pred_dirty430_cc \
    --gt-dir "../../Parotid-Project/Datasets/Dataset002_Parotid/labelsTs" \
    --model-name Custom_3D_UNet_dirty430_CC --results-csv ./eval.csv
```

---

## RESULTS (LOCKED 43-case test set, `eval_testset.py`, true anisotropic spacing)

| Arm | Train | Loss | 3D Dice | Tversky | HD95 (mm) | Surf-Dice | R Dice | L Dice |
|-----|-------|------|---------|---------|-----------|-----------|--------|--------|
| clean-208 (E4, raw) | 208 | Dice+BCE | 0.7681 | 0.7746 | 26.76 | 0.8383 | 0.799 | 0.737 |
| **clean-208 (E4, +CC)** | 208 | Dice+BCE | **0.7750** | 0.7788 | 6.05 | 0.8539 | — | — |
| dirty-430 (raw) | 430 | Dice+BCE | 0.7111 | 0.7164 | 47.63 | 0.7684 | — | — |
| **dirty-430 (+CC)** | 430 | Dice+BCE | **0.7296** | 0.7267 | 9.61 | 0.7999 | 0.745 | 0.715 |
| masked-430 (raw) | 430 | Masked | 0.7079 | 0.7401 | 134.69 | 0.7290 | 0.725 | 0.691 |
| **masked-430 (+CC)** | 430 | Masked | **0.7589** | 0.7746 | 9.19 | 0.8403 | 0.777 | 0.740 |
| _ref:_ nnU-Net 3d 1 fold | 208 | — | 0.8187 | 0.8166 | 5.25 | 0.8965 | — | — |
| _ref:_ nnU-Net dirty-430 (E2) | 430 | — | 0.7726 | 0.7644 | 5.45 | 0.8400 | — | — |

**Per-side (+CC):** masked-430 beats dirty-430 on **both** glands (R 0.777 vs 0.745, L 0.740 vs 0.715), and its
**L side (0.740) essentially matches clean-208's L (0.737)** — the whole masked−clean gap sits on the R side
(0.777 vs 0.799). So masking lifts both sides over the naive control and closes the left-gland gap to clean
entirely; it's the right gland where it still trails. (clean-208 E4: R 0.799 / L 0.737.)

Config: same as E4 — UNet3D 16.54M params, patch [48,224,192], batch 2, Adam 1e-4, AMP, seed 42, nnU-Net
fold-0 (344 train / 86 val). dirty-430 early-stopped @75 (best val-loss 0.2030 @ ep45); masked-430 trained
stably, best val-loss 0.1731 @ ep62. **Val losses are NOT comparable across arms** (masked ignores
un-annotated channels; ordinary penalises them) — only the test-set Dice above is the arbiter.

**Noise floor:** n=43, honest cross-run paired std ≈ 0.05 (master §22.2). masked−dirty (+0.029) and
masked−clean (−0.016) are both within/near it → suggestive, not definitive. Single fold, single seed.

## MECHANISM VERDICT

**Partial confirmation.** Two things held and one did not:

1. **masked-430 > dirty-430 (+0.029 CC): ✅ direction confirmed.** Masking the un-annotated gland out of the
   loss beats the ordinary loss that penalises it, on the same 430 patients. The masked model produces good
   gland cores but more stray false-positive islands (raw HD95 134.69 → 9.19 after largest-CC, a +0.051 Dice
   CC gain vs dirty's +0.019), so it *especially* wants the CC postproc E4 established.
2. **masked-430 (0.7589) < clean-208 (0.7750) < nnU-Net (0.8187): ✗ the strong prediction failed.** Adding
   the 222 masked single-side patients did **not** exceed training on the 208 clean both-annotated cases, let
   alone beat nnU-Net. §21's number-backed prediction ("should land above 0.8187") does **not** hold here.
3. **Secondary observation — masking trained more stably.** The dirty arm diverged at epoch 48 (train loss
   collapsed, val loss stuck at ~0.47 for 27 epochs; best checkpoint predates it at ep45). The masked arm
   never diverged (smooth to best @ ep62). This *may* be a real benefit of masking — the ordinary loss's
   conflicting gradients (punishing correct anatomy on un-annotated glands) are a plausible cause of the
   instability — or single-seed luck. It also means part of the +0.029 masked-over-dirty edge could reflect
   dirty's undertrained best checkpoint, not the loss alone. One seed cannot separate these.

## WHY THE STRONG PREDICTION LIKELY FAILED (hypotheses, not proven)
- **The test set is the wrong instrument.** All 43 cases are both-parotid QC-clean — the masked loss's main
  expected benefit is on **single-side** cases (learning the healthy gland without penalty), which this test
  set contains none of. It structurally can't reward the fix. (§20 limitation 8.)
- **Data-mix dilution.** With masking, the 222 single-side patients contribute only one gland each, so the
  model sees proportionally fewer *both-gland* examples than clean-208 does per patient — plausibly worse for
  a both-gland test set.
- **No training machinery.** Like E4, this plain U-Net omits nnU-Net's deep supervision / augmentation /
  schedule; the −0.044 E4↔nnU-Net residual is unrelated to the loss and caps how high any arm here can go.

## CAVEATS
- Single fold, single seed (matches E4); no seed-variance estimate.
- **The dirty-430 control diverged mid-training** (best-val checkpoint @ ep45, before the epoch-48 collapse),
  so masked-vs-dirty is confounded by training stability, not only the loss. Re-running dirty with gradient
  clipping / a different seed would tighten this.
- The 43-case test set is the both-parotid QC-clean easy subset — it does **not** measure single-side
  performance, which is exactly where the masked loss should help most (§20 limitation 8).
- Fold-0 split for the 430 set is the reproduced nnU-Net `KFold(5, seed=12345)`; `splits_final.json` for
  Dataset003 did not survive to the Mac, but the reproduction is faithful by construction (E4 validated it).
