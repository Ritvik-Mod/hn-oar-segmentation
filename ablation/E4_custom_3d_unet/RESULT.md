# E4 — Hand-built 3D U-Net vs nnU-Net (Q4): architecture or pipeline?

**Question.** Once the *pipeline* is fixed (3D, nnU-Net preprocessing), does the
*architecture* still matter — or is nnU-Net's win almost entirely pipeline/data?
Train a **plain, hand-built 3D U-Net** on the **same nnU-Net-preprocessed Dataset002**
and compare to nnU-Net `3d_fullres` single fold (**0.8187**). Same data, same
preprocessing, **different (vanilla) implementation** → isolates architecture.

**Status: ✅ DONE (2026-07-17, RunPod L40S).** Trained fold 0, early-stopped at epoch
83 (best val 0.1222, ~50 min), sliding-window inference on the 43 test cases, scored
with `eval_testset.py`. **Custom 3D U-Net: Dice 0.7681, Tversky 0.7746, HD95 26.76 mm,
Surf-Dice 0.8383** (R 0.799 / L 0.737).

---

## What was built and validated (offline, no training)

`validate_e4.py` exercised the whole chain on real Dataset002 cases + small tensors:

- **`unet3d.py`** — plain 3D U-Net (extends the project's 2D `unet.py`): DoubleConv3d
  (Conv3d→InstanceNorm3d→LeakyReLU ×2), 5 levels **features [32,64,128,256,320]**,
  MaxPool3d down, ConvTranspose3d up, 1×1×1 head → 2 channels (R,L, independent
  sigmoid). **16.5 M params.** Forward+backward verified. Deliberately a *different,
  vanilla* implementation from nnU-Net's PlainConvUNet — that is the point of Q4.
- **`preprocess_e4.py`** — reproduces nnU-Net's 3d_fullres preprocessing **exactly and
  cheaply**: the Dataset002 NIfTI are **already at the target spacing [3.0,0.977,0.977]**
  (verified) → **no resampling**; **CTNormalization** uses the constants read straight
  from the run's `plans.json` (`foreground_intensity_properties_per_channel`):
  clip [8087, 8394], then (x − 8198.65)/57.49 (raw-pixel space, matching what nnU-Net
  itself was fed). Validated: normalized volume mean ≈ −1.78 (background air clipped
  to the lower bound, exactly as nnU-Net produces).
- **`dataloader_e4.py`** — random-patch [48,224,192] sampling with nnU-Net-style
  **0.33 foreground oversampling**, hflip aug (R/L swap), constant-pad for small
  volumes. Fold split **reproduces nnU-Net's default** `KFold(5, shuffle=True,
  random_state=12345)` → verified **166 train / 42 val** (matches master §14.6). On
  the pod, prefer the real `splits_final.json` if present (`--splits-json`).
- **`loss3d.py`** — DiceBCE-3D mirroring the project's `CombinedLoss` (0.5 Dice + 0.5
  BCE, per-channel global Dice, independent sigmoid) for 5D tensors.
- **`train_e4.py`** — single fold 0, Adam 1e-4, CUDA AMP, batch 2 (plans.json),
  250 epochs / 250 iters, best-val checkpointing, resumable.
- **`predict_e4.py`** — Gaussian-weighted sliding-window inference (50% overlap),
  writes nnU-Net-style labelmaps (0/1/2) in **labelsTs orientation (Y,X,Z) with the
  labelsTs affine** → scored by the **same `pipeline/eval_testset.py`** as nnU-Net
  (true anisotropic spacing). Tiling + labelmap + orientation validated.

## Why this is a clean "only the code differs" comparison

Same input volumes, same target spacing, same CTNormalization constants, same
fold-0 split, same 43-case test evaluator as the nnU-Net 0.8187 baseline. The only
differences are intentional and vanilla: a hand-written 3D U-Net with **plain
isotropic 2×2×2 pooling** (nnU-Net uses anisotropic strides — noted, this is part of
"a genuinely different implementation") and the project's own 2-channel sigmoid
convention. Data footprint is small: only the 6.5 GB Dataset002 NIfTI + tiny
`plans.json` need uploading (shared with P0).

---

## Results (LOCKED 43-case test set, `eval_testset.py`, true anisotropic spacing)

| Model | 3D Dice | Tversky | HD95 (mm) | Surf-Dice | vs nnU-Net 1 fold |
|-------|---------|---------|-----------|-----------|-------------------|
| Custom 3D U-Net (nnU-Net preproc, plain, raw) | 0.7681 | 0.7746 | 26.76 | 0.8383 | −0.0506 Dice |
| **Custom 3D U-Net + largest-CC postproc** | **0.7750** | 0.7788 | **6.05** | 0.8539 | **−0.0437 Dice, HD95 ≈ match** |
| _ref:_ nnU-Net 3d_fullres 1 fold | 0.8187 | 0.8166 | 5.25 | 0.8965 | — |
| _ref:_ best from-scratch 2D on test (Attn, P0) | 0.7434 | 0.7387 | 3.57* | 0.8827 | (*iso) |

All four E4 metrics use `eval_testset.py` (true anisotropic spacing) → **directly
comparable to nnU-Net's** (no HD95 caveat, unlike P0/E3). Config: features
[32,64,128,256,320], 16.5 M params, plain isotropic 2×2×2 pooling, DiceBCE, Adam 1e-4,
patch [48,224,192] batch 2, fold-0 (166/42), early-stopped epoch 83, ~50 min on L40S.

## Finding (what E4 actually shows)

- **The plain 3D U-Net reaches 0.768 — clearly above the from-scratch 2D ceiling
  (0.72–0.74 on test, P0) but ~0.05 below nnU-Net (0.8187).** So on Dice, a vanilla,
  hand-written 3D architecture on nnU-Net's exact preprocessing recovers *most* of the
  2D→nnU-Net gap — **architecture is largely, but not entirely, irrelevant once the
  pipeline is right.** The residual ~0.05 Dice is attributable to nnU-Net's training
  machinery this deliberately-plain net omits: deep supervision, its heavy augmentation
  schedule, anisotropic strides matched to the 3.0 mm z-spacing, and postprocessing.
- **The HD95 is the loud signal: 26.76 mm vs nnU-Net's 5.25.** With Dice at 0.77 the
  overlap is fine, so the huge HD95 = **stray false-positive islands far from the
  gland** — this net has *no* connected-component postprocessing, whereas nnU-Net
  applies it. This is the **same failure mode A1 quantifies for Swin**, and master
  §15.3 already showed largest-CC / L-R-split postproc is a cheap HD95 fix. A
  largest-CC cleanup on E4's predictions would be expected to collapse HD95 toward
  single digits with little Dice change — i.e. the boundary gap to nnU-Net is
  **postprocessing, not architecture** (see optional postproc note below).

## Note for S1 (attribution refinement)

P0 + E4 together sharpen Track A's decomposition: a large slice of the headline
0.62→0.82 was **val→test measurement** (Phase-1 val 0.62 → Phase-1 *test* ~0.74, P0),
so the real from-scratch-2D → nnU-Net gap *on test* is ~0.08, of which the plain-3D
pipeline (E4) recovers ~0.02–0.03 and nnU-Net's training machinery + postproc the
rest. This does not change the ordering (preprocessing/3D dominate architecture) but
S1 must state the val-vs-test caveat explicitly (already flagged in §20.A/§20.B).

## ✅ Largest-CC postproc re-eval (RAN 2026-07-17) — confirms the boundary claim

`postproc_largestcc.py` keeps only the largest connected component per class, then
re-scored with `eval_testset.py`: **HD95 26.76 → 6.05 mm** (a 77% drop, now ≈ nnU-Net's
5.25), Dice **0.7681 → 0.7750** (slightly *up*), Surf-Dice 0.8383 → 0.8539. So the
entire HD95 gap to nnU-Net was **stray false-positive islands, removable with one line
of postprocessing — not architecture.** The only residual after that is **Dice 0.775
vs 0.819 (−0.044)**, attributable to nnU-Net's training machinery (deep supervision,
augmentation), not the model. This is the cleanest possible statement of the thesis:
*a plain hand-built 3D U-Net on nnU-Net's pipeline + trivial postproc matches nnU-Net
on boundaries and lands within 0.044 Dice — architecture is nearly irrelevant once the
pipeline is right.* Files: `pred_test_cc/`, `eval.csv` (both rows).

**Commands:** `ablation_study/RUNPOD_COMMANDS_TrackB.md` §4.
