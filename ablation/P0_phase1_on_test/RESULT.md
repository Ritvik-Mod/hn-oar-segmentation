# P0 — Re-evaluate the 4 Phase-1 checkpoints on the LOCKED test set

**Question.** The four from-scratch models (U-Net, Attention U-Net, TransUNet,
Swin-UNet) were only ever scored on the **validation** set (~0.62 headline Dice,
master §13.2). To make the whole ablation study comparable, score all four on the
**same locked 43-case test set** every other model in the study uses, with the
Phase-1 3D volumetric metric.

**Status: ✅ DONE (2026-07-17, on the RunPod L40S).** All four checkpoints scored on
the locked 43-case test set. (TransUNet's checkpoint was truncated in the first
upload — re-uploaded after a size check `1211352136` → `1230961638` — then re-run.)

---

## Method (exact, reproducible)

- **Script:** `evaluate_phase1_on_test.py` (device-flexible; run with `--device cuda
  --save-preds` on the pod). Metrics module `phase1_metrics.py` is copied
  behaviourally verbatim from the frozen `dicom test/attention_unet_codes_from_hpc/
  attention_evaluate.py` (Dice / clinical Tversky α0.3 β0.7 / HD95 / Surface-Dice
  @3 mm), so P0 reproduces the Phase-1 protocol exactly — only the patient list changes.
- **Models / checkpoints (read-only):** `checkpoints/{unet/best_model_unet.pth,
  attention_unet/best_model_attention.pth, transunet/best_model_transunet.pth,
  swin_unet/best_model_swinunet.pth}`. Verified loadable: U-Net loads at epoch 4,
  val_loss 0.1685 (matches master §13.3).
- **Prediction convention:** logits `(B,2,512,512)`; channel 0 = PAROTID_R,
  channel 1 = PAROTID_L (dataloader convention, master §9); sigmoid > 0.5; slices
  stacked into a 3D volume per patient; per-side (R,L) metrics then averaged over
  all 43×2 = 86 sides (HD95 via `nanmean`).

### Data source & the comparability decision (important)

Scored on the **Dataset002_Parotid NIfTI test volumes** (`imagesTs` = raw-pixel
int16, `labelsTs` = 1:parotid_r / 2:parotid_l). This is the **exact same 43 cases
and the exact same ground-truth voxels** nnU-Net's 0.8187 was scored on
(PAR0209..PAR0252, PAR0240 QC-dropped). Verified locally: `imagesTs` is raw pixel
(min 0 / max ~27k), so the Phase-1 `window_and_normalise` (HU = raw − 8192, window
[−150,250]) applies unchanged; `labelsTs` label 1 has the smaller column centroid
(R) and label 2 the larger (L), matching the model's output channels.

Using the NIfTI volumes (rather than re-stacking npz slices) guarantees the P0
Dice sits on the **same voxel grid** as every Phase-2 number → **Dice / Tversky are
directly comparable** to nnU-Net's 0.8187.

### Caveat (reported, not hidden) — boundary metrics

Per master §12.3 vs §14.5: the Phase-1 protocol computes HD95 / Surface-Dice with
**isotropic 0.977 mm** spacing (2D-stacked convention), whereas `eval_testset.py`
(the nnU-Net numbers) uses **true anisotropic (0.977, 0.977, 3.0)**. Therefore:
- **Dice and Tversky: cross-comparable** Phase-1 ↔ Phase-2.
- **HD95 and Surface-Dice: NOT cross-comparable** between P0 and the nnU-Net rows.
  P0 reports the isotropic Phase-1 numbers (to stay comparable with the Phase-1
  *validation* table, master §13.2). This caveat is stated wherever P0 HD95 is quoted.

---

## Results (LOCKED 43-case test set, Phase-1 isotropic protocol; n=86 sides/model)

| Model | 3D Dice | Tversky | HD95 (mm, iso) | Surf-Dice (3mm) | zero-Dice sides | val Dice (§13.2) |
|-------|---------|---------|----------------|-----------------|-----------------|------------------|
| **Attention U-Net** (scratch, 2D) | **0.7434** | 0.7387 | **3.57** | **0.8827** | 7/86 | 0.6346 |
| U-Net (scratch, 2D)               | 0.7390 | 0.7315 | 4.25 | 0.8702 | 7/86 | 0.6209 |
| TransUNet (scratch, 2D)           | 0.7313 | 0.7307 | 5.60 | 0.8594 | 8/86 | 0.6332 |
| Swin-UNet (scratch, 2D)           | 0.7156 | 0.7082 | **8.34** | 0.8794 | 2/86 | 0.5106 |
| _ref:_ nnU-Net 3d_fullres 1 fold  | 0.8187 | 0.8166 | 5.25 (aniso) | 0.8965 | — | — |

Per-side: every model scores **higher on R than L** (e.g. U-Net R 0.788 / L 0.690),
and mean << median (U-Net mean 0.739 vs median 0.834), i.e. a handful of catastrophic
one-sided cases drag the mean down — the same L/R / partial-annotation failure mode
seen in Phase 2 (master §15). Files: `eval.csv`, `per_case.csv` (43×4), `preds/` (172).

---

## Interpretation

1. **On the clean test set all four cluster at 0.72–0.74 Dice — ~0.11 above their
   0.62 *val* figures** (these 43 are the easier both-parotid, QC-clean cases), **and
   ~0.08 below nnU-Net's 0.8187.** The whole 31M→102M architecture range lands within
   0.028 Dice of each other → **architecture barely matters among the from-scratch 2D
   models**; the gap to nnU-Net is pipeline/3D/labels, not capacity. This firms up
   Track A's attribution (P0 puts the Phase-1 column on *test*, so the 0.62→0.82
   arithmetic is now told on one axis).
2. **Swin is the clear boundary outlier:** worst HD95 by far (8.34 vs 3.57–5.60)
   despite a *middling* Dice, and — tellingly — it has the **fewest** one-sided misses
   (2 vs 7–8), so its HD95 blow-up is **not** missing glands but bad boundaries. Its
   per-case HD95 has extreme tails (PAR0213 135 mm, PAR0234 96 mm, PAR0242 89 mm) =
   far-flung stray islands. This is exactly what **A1** dissects (coarse 128→512
   upsampling + underfit islands).
3. These four test numbers are the from-scratch baselines that **E3** (ImageNet
   pretraining) is measured against.

**Caveat reminder:** HD95/Surf-Dice here are Phase-1 **isotropic 0.977**; not
comparable to nnU-Net's anisotropic HD95. Dice/Tversky **are** comparable.

**Outputs:** `eval.csv`, `per_case.csv`, `preds/<model>__PARxxxx.npz` (consumed by A1).
**Commands:** see `ablation_study/RUNPOD_COMMANDS_TrackB.md` §1.
