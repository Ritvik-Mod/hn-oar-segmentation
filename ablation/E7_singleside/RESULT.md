# E7 Part 1 — Single-side evaluation

**Question:** every headline number in the study is on the 43 both-parotid QC-clean test cases (the "easy"
subset). How do the models perform on the **single-side** TEST-split patients — the realistic, clinically
common cases that were **never trained on** — on the annotated (at-risk) side, and do the models also predict
the un-annotated healthy gland (the annotation-gap suppression signal, master §16)?

Status: **✅ DONE (2026-07-18).** Prep + validation done locally; all inference run on the Part-2 pod (2× A40);
scoring + montage produced on-pod, downloaded to `../E7_experts/_pod_results/`. Results in §7 below.

---

## 1. Cohort — verified from raw data (not prose)

Derived TEST-split parotid presence with `scan_test_presence.py` (reuses E2's exact CT-group selection rule;
reads per-organ masks from `ML_Dataset_Final/<pid>/**/*.npz`). Output: `parotid_presence_test.csv`.

| TEST split (165) | both | only_R | only_L | none |
|---|---|---|---|---|
| count | **44** | **30** | **28** | 63 |

- **Single-side = 58** (30 only_R + 28 only_L) — exactly matches the task doc's prediction (≈30+28).
- The 44 "both" become the **43** QC-clean cases in `Dataset002/labelsTs` (one dropped by the 3×-median QC rule).
- **Leakage check:** all 58 are TEST-split; the presence CSV that defines every model's TRAIN/VAL pool is
  `E2/parotid_presence_trainval.csv` (667 rows = train 583 + val 84), a disjoint set. No single-side test
  patient was trained on by any model. ✅

## 2. L/R naming reliability (single-side validity check)

`fix_labels_qc.relabel_geometric` cannot correct L/R on a single-gland case (it returns early when one side is
empty), so single-side labels keep whatever the clinician named them. Checked clinician `status` vs **geometry**
(annotated gland's mean image column vs the 255.5 midline; from the both-parotid cases, patient-right glands sit
at *lower* columns, patient-left at *higher*):

- **58/58 agree.** only_R glands cluster at columns **177–221**; only_L at **291–325** — a clean gap (221 vs
  291), no ambiguity. The single-side test labels are trustworthy and already on the geometric convention every
  model learned. (GT labels are still assigned geometrically in the build, a no-op here.)

## 3. Built volumes (compact, ready to upload to the pod)

`build_singleside.py` reconstructs each of the 58 patients with `pipeline/build_volumes.reconstruct_patient`
(identical 0.977×0.977×Z affine) and the E4/E6 `norm_constants.json` verbatim, so single-side cases are
preprocessed **exactly** like the clean-43 every model was scored on. All 58 are z=3.0 mm, (512,512,~120) —
identical geometry to Dataset003. Outputs:

- `nnraw/imagesTs/` + `nnraw/labelsTs/` (58 each) — nnU-Net raw NIfTI (image + single-foreground GT labelmap,
  0/1/2). **1.5 GB** → for the nnU-Net clean-208 predict.
- `e6npz/` (58 `.npz`: CTNormalized `data` f16 (Z,Y,X), `seg`, `mask`) — **228 MB** → for the custom E4/E6 predicts.
- `case_map.json` — case → {patient, status, label, centroid_col, gland_vox}. Result: 30 R-side, 28 L-side.

## 4. Harness validated

`pipeline/eval_testset.py` reproduces the **nnU-Net clean-208 baseline = 0.8187** exactly (R 0.8278 / L 0.8095)
on the existing `pred_nomirror` clean-43 predictions. Metric code is trusted.

`eval_testset.py` already skips any side with zero GT voxels, so run on single-side GT it scores **only the
annotated side** — exactly the clinical question. `eval_singleside.py` wraps this to also split by side and
measure contralateral behaviour on **raw** predictions (largest-CC-per-class keeps contralateral blobs, so raw
is the correct basis for the suppression figure).

## 5. Models to evaluate (checkpoints located, all on disk)

| arm | checkpoint | clean-43 baseline |
|---|---|---|
| nnU-Net clean-208 | `Parotid-Project/.../nnunet/trained_model_v2_noMirror_FINAL/` (fold_0, checkpoint_final) | 0.8187 |
| E4 custom clean-208 | `_pod_results_B/checkpoints/E4_custom_3d_unet_best.pth` | 0.7681 / 0.7750 +CC |
| E6 dirty-430 | `_pod_results_B/E6_masked_loss/ckpt_dirty430/best_model.pth` | 0.7296 +CC |
| E6 masked-430 | `_pod_results_B/E6_masked_loss/ckpt_masked430/best_model.pth` | 0.7589 +CC |
| (+ Part-2 experts, if trained) | — | — |

## 6. How it was run (on the 2× A40 pod, batched with Part 2)

`predict_nnunet.py` (nnU-Net clean-208, mirroring OFF) + `predict_e6.py` (e4 / dirty430 / masked430, cuda) on
the 58, plus the two experts combined → `experts_ss`. Scored with `eval_singleside.py`, montage by
`make_montage.py`. Outputs in `../E7_experts/_pod_results/{preds,results}/`.

## 7. RESULTS — annotated-side accuracy (n=58 single-side)

Dice+CC on the ONE annotated (at-risk) gland; skips the un-annotated side (fair). Split by side (R=30, L=28).

| model | Dice+CC | Tversky | HD95 | SurfDice | R | L |
|---|---|---|---|---|---|---|
| **E7 experts** | **0.8552** | 0.8456 | 4.07 | 0.9328 | 0.8449 | 0.8663 |
| nnU-Net clean-208 | 0.8523 | 0.8421 | 3.91 | 0.9325 | 0.8445 | 0.8607 |
| E4 custom clean-208 | 0.8011 | 0.7950 | 4.74 | 0.8847 | 0.7902 | 0.8128 |
| E6 masked-430 | 0.8010 | 0.8096 | 4.95 | 0.8820 | 0.7853 | 0.8179 |
| E6 dirty-430 | 0.7436 | 0.7306 | 13.01 | 0.8139 | 0.6961 | 0.7944 |

- **The at-risk gland is segmented WELL on single-side (~0.85 for the best two).** Notably *higher per-gland*
  than the both-parotid average (nnU-Net 0.8523 single vs 0.8187 both; experts 0.8552 vs 0.8099). The
  single-side annotated gland is the *easier* target (bigger/clearer), NOT the catastrophe the clinical framing
  feared. This is the key Part-1 finding: **the models do not fail on the realistic single-side cases.**
- Experts edge nnU-Net for the top spot; the nnU-Net-based arms (experts, clean-208) clearly beat the custom
  U-Net arms (E4/E6) — an architecture effect, consistent with the both-parotid ordering.
- E6 **dirty-430 collapses on boundary quality** (HD95 13.0 mm — stray islands); masked-430's raw→+CC jump
  (0.7533→0.8010) shows it too had islands the largest-CC cleanup removed.

## 8. RESULTS — contralateral-prediction behaviour (the annotation-gap signal, raw preds)

Does each model *also* segment the un-annotated healthy gland? Per master §16 this is anatomically CORRECT, not
an error. "Predicts contralateral" = largest contra blob ≥ 500 vox AND ≥ 20 % of the ipsilateral prediction.

| model | mean contra vox | median blob | contra-rate |
|---|---|---|---|
| E4 custom clean-208 | 5672 | 5323 | 100.0% |
| E7 experts | 4577 | 3963 | 98.3% |
| nnU-Net clean-208 | 4149 | 3556 | 96.6% |
| E6 masked-430 | 7010 | 5650 | 94.8% |
| **E6 dirty-430** | 5816 | 5135 | **87.9%** ← most suppression |

- **The gap effect is directionally confirmed but MODEST.** dirty-430 (penalise the un-annotated side as
  background) predicts the healthy gland in the fewest cases (87.9 %), and masking recovers it (94.8 %) — exactly
  the predicted ordering. But *all* models predict the contralateral gland in ≥88 % of cases, so the annotation
  gap does **not** strongly suppress the healthy gland at inference. §16's suppression is real but small here.
- Median contra blobs are 3.5k–5.6k vox = genuine gland-sized predictions, so scoring "annotated side only" is
  fair (we are not penalising the correct contralateral predictions).

## 9. Montage
`../E7_experts/_pod_results/results/contralateral_montage.png` — 4 single-side cases (2 only_R, 2 only_L), GT vs
each model, green = annotated (at-risk) side, red = contralateral gland the model also predicts.

## Scripts (this dir)
`scan_test_presence.py` · `build_singleside.py` · `predict_nnunet.py` · `predict_e6.py` (E6 copy) ·
`eval_singleside.py` · `make_montage.py`. Seed n/a (deterministic inference). Cost: shared with Part 2 (§7 of
`../E7_experts/RESULT.md`) — no separate spend.
