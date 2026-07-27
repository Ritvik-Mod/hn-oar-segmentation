# TRACK A — nnU-Net experiments (Claude Code Account #1)

> You are **Track A**. You own **E1, E2, E5**. All of these run inside nnU-Net; none of them need anything
> from Track B. First read `AGENT_INSTRUCTIONS.md`, then `ABLATION_PLAN.md`, with `MASTER_PROJECT_REFERENCE.md`
> as context. Follow the golden rules (locked test set, no-mirror trainer, `--disable_tta`, read-only zones).
>
> Output everything under `ablation_study/E1_2d_vs_3d/`, `ablation_study/E2_annotation_gap/`,
> `ablation_study/E5_ensembling/`. Keep the status table + master §20.A live as you go.

## What you have to start from
- The prepared nnU-Net dataset **`Dataset002_Parotid`** (208 train + 43 test, L/R-corrected, QC-clean).
  Local copy layout and locations are in the master doc §14 / `HANDOFF_5fold_ensemble.md`.
- The custom trainer **`nnUNetTrainer_250epochs_noMirror`** (`pipeline/nnUNetTrainer_250epochs_noMirror.py`).
- The evaluator **`pipeline/eval_testset.py`** (`--pred-dir`, `--gt-dir`, `--model-name`).
- The existing **`3d_fullres` single-fold (Dice 0.8187)** and **5-fold ensemble (0.8202)** results — your
  baselines to compare against.

## Environment (every fresh pod)
Set `nnUNet_raw / nnUNet_preprocessed / nnUNet_results`; `pip install nnunetv2 nibabel scipy scikit-image`;
**register the custom trainer** into the nnU-Net package (master §14.4 / handoff STEP 2). Prefer a cheap GPU
(4090/A5000). Run trainings in `tmux`.

---

## E1 — 2D vs 3D  *(isolates dimensionality, Q1)*
Dataset002 is already planned/preprocessed for nnU-Net. Ensure the **`2d`** configuration is preprocessed
(re-run `nnUNetv2_plan_and_preprocess -d 2 -c 2d` if needed), then:
```bash
# train the 2d config, fold 0, no-mirror trainer
nnUNetv2_train 2 2d 0 -tr nnUNetTrainer_250epochs_noMirror --npz
# predict the locked test set (mirroring off)
nnUNetv2_predict -i $nnUNet_raw/Dataset002_Parotid/imagesTs -o ablation_study/E1_2d_vs_3d/pred \
  -d 2 -c 2d -tr nnUNetTrainer_250epochs_noMirror -f 0 --disable_tta
# evaluate on the SAME test labels
python pipeline/eval_testset.py --pred-dir ablation_study/E1_2d_vs_3d/pred \
  --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs --model-name nnUNet_2d_fold0 \
  --results-csv ablation_study/E1_2d_vs_3d/eval.csv
```
- **Compare to:** `3d_fullres` single-fold (0.8187). Same data, same everything except 2D vs 3D.
- **Write:** `ablation_study/E1_2d_vs_3d/RESULT.md` (question, commands, seed/GPU/time, metric table, and
  interpretation: does 2D land near 0.75+ → 3D buys little → ceiling was preprocessing/labels; or collapse
  → 3D is the driver).

## E2 — Annotation gap: clean vs dirty labels  *(isolates label quality, Q2 — the key one)*
Build a **"dirty" training dataset** that reproduces the baseline's annotation-gap noise, but **keep the
test set identical to the clean Dataset002 test set.**
1. Copy `pipeline/make_nnunet_dataset.py` → `ablation_study/E2_annotation_gap/make_dirty_dataset.py` and
   modify it: **include every TRAIN/VAL patient that has ≥1 parotid** (not just both-annotated); write each
   patient's labels **as-is** (single-side patients keep only their annotated side; the un-contoured side is
   left as background). Keep the QC volume-filter (drop >3× median). Assign it a new id, e.g.
   **`Dataset003_ParotidDirty`**. For `imagesTs/labelsTs`, **copy the exact clean Dataset002 test files**
   (do not rebuild the test set).
2. `nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity`
3. Train + predict + evaluate exactly like E1 but with `-d 3 -c 3d_fullres`:
```bash
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror --npz
nnUNetv2_predict -i $nnUNet_raw/Dataset003_ParotidDirty/imagesTs -o ablation_study/E2_annotation_gap/pred \
  -d 3 -c 3d_fullres -tr nnUNetTrainer_250epochs_noMirror -f 0 --disable_tta
python pipeline/eval_testset.py --pred-dir ablation_study/E2_annotation_gap/pred \
  --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs --model-name nnUNet_3d_dirtylabels_fold0 \
  --results-csv ablation_study/E2_annotation_gap/eval.csv
```
- **Compare to:** clean Dataset002 `3d_fullres` single-fold (0.8187). Only the *training-label condition*
  differs; test is the same clean 43 cases.
- **Expected:** dirty-trained scores lower / misses more single-side cases → quantifies the annotation-gap
  cost. **This is the study's most original result.**
- **Write:** `ablation_study/E2_annotation_gap/RESULT.md` incl. how many extra (single-side) patients the
  dirty set added, and the metric delta vs clean.
- *(Optional, only if you have time AND the owner wants it: also flag that a `masked_loss.py`-trained variant
  on this same dirty set would show the fix — but that needs custom training, so leave it to Track B / a
  later task; just note it as future work in the RESULT.)*

## E5 — Ensembling delta  *(mostly bookkeeping)*
Single fold (0.8187) vs 5-fold ensemble (0.8202) are already computed in the main project results. Tabulate
the delta in `ablation_study/E5_ensembling/RESULT.md` and note it's small here. Only retrain folds if the
existing 5-fold results are genuinely missing (they shouldn't be — see `Parotid-Project/Results/`).

---

## When you finish
- All of E1/E2/E5 have a `RESULT.md`, filled status rows (E1/E2/E5) in `ABLATION_PLAN.md`, and appended
  entries under **master §20.A**. Add a "Track A complete" line under §20.A.
- Do **not** run S1 unless the status table shows Track B (E3/E4/A1/P0) also ✅.
