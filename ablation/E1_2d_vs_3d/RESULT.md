# E1 — Dimensionality: 2D vs 3D

**Track A · Status: ✅ DONE · Date: 2026-07-16**

## HEADLINE: 3D buys almost nothing — **+0.007 Dice** over 2D.

| Model | 3D Dice | Tversky | HD95 (mm) | Surf-Dice | Source |
|---|---|---|---|---|---|
| nnU-Net `3d_fullres`, 1 fold (baseline) | **0.8187** | 0.8166 | 5.25 | 0.8965 | `results_tracker_TEST.csv` |
| **nnU-Net `2d`, 1 fold (E1)** | **0.8117** | 0.8085 | 5.52 | 0.8943 | `eval_E1_2d.csv` |
| **Δ (2d − 3d)** | **−0.0070** | −0.0081 | +0.27 | −0.0022 | |

Everything except dimensionality was identical: same Dataset002, same `nnUNetTrainer_250epochs_noMirror`,
same fold 0, same locked 43-case test set, `--disable_tta`.

**The pre-registered hypothesis (below) said 2d would land 0.72–0.78. It beat that, at 0.8117.** The
prediction was directionally right but understated the effect: 3D is worth even less than expected.
Surface-Dice is effectively identical (0.8943 vs 0.8965), so 3D does not even buy boundary quality here.

**Interpretation.** The 2D→3D axis accounts for **~0.7 of the ~20 Dice points** in the 0.62 → 0.82 headline
— about 3.5% of the gain. Combined with E5 (ensembling, +0.0015, not significant) and E2b (annotation gap,
−0.1288), the attribution is now: **labels ≫ preprocessing ≫ 3D > ensembling**. The Phase-1 ceiling at 0.62
was **not** caused by working in 2D, which vindicates the Phase-1 diagnosis in master §13.4 ("the ceiling is
from the data/preprocessing, not architectural capacity") and extends it — it was not dimensionality either.

**Why this matters practically:** a 2D model at 0.8117 is far cheaper to train (26.4 s/epoch vs 35.3 s) and
to run, and needs no 3D context at inference. If the deliverable were cost-sensitive, 2D is nearly free
accuracy-wise. The 3D pipeline's value is in the *preprocessing* it brings, not the third dimension.

**Sanity:** internal fold-0 validation pseudo-Dice was **[0.8328, 0.8490]** — nowhere near the 0.4538 that
signals the L/R mirror bug (master §15.1), confirming the no-mirror trainer registered and worked.

---

## Original pre-registration (kept for the record)

---

## Question

Is the ~0.62 Phase-1 ceiling caused by **2D vs 3D**, or by preprocessing/labels? How much does 3D context
alone actually buy (Q1)?

Train nnU-Net's **`2d`** configuration on the same Dataset002, same no-mirror trainer, same single fold,
same locked test set — everything identical to the existing `3d_fullres` single-fold run (0.8187) *except*
dimensionality.

## Status

**Data is ready; only the GPU run is pending.** The clean `Dataset002_Parotid` raw dataset is present locally
at `Parotid-Project/Datasets/Dataset002_Parotid` (6.5 GB, 208 train + 43 test, verified: 251 kept / PAR0240
dropped / 1 L/R flip — matching master §14.3). E1 needs **no new dataset** — it trains the `2d` configuration
on that same clean dataset.

*(Correction: an initial `find -maxdepth 4` missed this dataset, which sits at depth 6. Master §18 also
describes `Parotid-Project/Datasets/` as holding only `dataset.json` + `case_mapping*.json` — that is stale;
it holds the complete volumes. §18 is frozen, so the correction is recorded here and in §20.A.)*

**Not run locally — no CUDA GPU.** No partial or substitute result is reported, because any downscaled
variant (fewer epochs, smaller patch, different test set) would not be comparable to the 0.8187 baseline and
would therefore answer a different question than Q1. Per the golden rules, scope is unchanged and the blocker
is logged rather than worked around.

**The owner is renting a RunPod pod.** E1 is covered by the command sheet at
`ablation_study/RUNPOD_COMMANDS_TrackA.md` (upload → trainer registration → `-d 2 -c 2d` preprocess → train →
predict `--disable_tta` → eval). Estimated ~1–2 h, ~$1–2 on a 4090.

## Blocker detail (⚠️)

**No CUDA GPU.** The local Mac has MPS only — Apple unified memory, **8 GB**, 8 CPU cores;
`torch.cuda.is_available() == False` (torch 2.12.1). Measured on this device:

```
encoder-only fwd+bwd, batch 4 @ 512x512: 247 ms/step
```

nnU-Net trains **250 epochs × 250 iterations**. Scaling that measurement to nnU-Net's real `2d` config
(add the decoder ≈ 2×, batch 12 vs 4 ≈ 3×) gives a **floor of ~26 hours**, and that floor ignores deep
supervision, the augmentation pipeline (CPU-bound on 8 cores), and dataloader overhead — realistically
**2–4 days**, with a serious risk of OOM at batch 12 / 512×512 in 8 GB of memory shared with the OS.
nnU-Net's MPS backend is also not a well-tested path. For reference, `3d_fullres` (patch [48, 224, 192],
batch 2) is far heavier still and is simply not viable here.

The task file's own guidance ("Prefer a cheap GPU (4090/A5000). Run trainings in `tmux`") assumes a rented
cloud GPU; that matches the project's history (master §14.8 — nnU-Net was never trained on HPC due to GPU
contention, hence RunPod).

## To run on the pod (full sheet: `ablation_study/RUNPOD_COMMANDS_TrackA.md`)

```bash
# 1. ensure the 2d config is preprocessed (Dataset002 is uploaded as-is; no rebuild needed)
nnUNetv2_plan_and_preprocess -d 2 -c 2d

# 2. train the 2d config, fold 0, no-mirror trainer (register the trainer first — master §14.4)
nnUNetv2_train 2 2d 0 -tr nnUNetTrainer_250epochs_noMirror --npz

# 3. predict the locked test set (mirroring off)
nnUNetv2_predict -i $nnUNet_raw/Dataset002_Parotid/imagesTs -o ablation_study/E1_2d_vs_3d/pred \
    -d 2 -c 2d -tr nnUNetTrainer_250epochs_noMirror -f 0 --disable_tta

# 4. evaluate on the SAME test labels
python3 pipeline/eval_testset.py --pred-dir ablation_study/E1_2d_vs_3d/pred \
    --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs --model-name nnUNet_2d_fold0 \
    --results-csv ablation_study/E1_2d_vs_3d/eval.csv
```

Expected cost: ~1–2 h on a 4090 (~$1–2). Record seed, GPU type, and wall-time from the nnU-Net training log
(`nnUNet_results/.../fold_0/training_log_*.txt`) into this file when run.

## Actual run record (reproducibility)

| Item | Value |
|---|---|
| Command | `nnUNetv2_train 2 2d 0 -tr nnUNetTrainer_250epochs_noMirror --npz` |
| Predict | `nnUNetv2_predict -i .../Dataset002_Parotid/imagesTs -d 2 -c 2d -tr nnUNetTrainer_250epochs_noMirror -f 0 --disable_tta` |
| Eval | `eval_testset.py --gt-dir .../Dataset002_Parotid/labelsTs --model-name nnUNet_2d_fold0` |
| GPU | 1× **NVIDIA L40S** (46 GB), on a 3× L40S RunPod box (E1 on GPU 1) |
| Epoch time | **26.4 s** × 250 epochs ≈ **110 min** |
| Seed | nnU-Net default (deterministic per fold; see `fold_0/debug.json`) |
| Config | `2d`: patch [512, 512], batch 12, 8 stages [32,64,128,256,512,512,512,512] — verified identical to Dataset002's original plan |
| Date | 2026-07-16, finished 19:11 |
| Cost | ~$1 (share of a $2.97/hr 3-GPU box) |
| Artifacts | `_pod_results/results/eval_E1_2d.csv`, `_pod_results/preds/E1_2d/` (43 predictions), `_pod_results/nnUNet_results/Dataset002_Parotid/.../2d/fold_0/` (checkpoint, `progress.png`, training log) |

## Hypothesis (recorded BEFORE the run — outcome noted above)

Stated up front so the result cannot be rationalised after the fact:

- **If `2d` lands ≥ ~0.75** → 3D buys only a few points, and the Phase-1 ceiling was **preprocessing +
  labels**, not dimensionality. Combined with E5's null ensembling result, that would leave preprocessing and
  label quality as the sole drivers of 0.62 → 0.82.
- **If `2d` collapses toward ~0.62** → 3D context is the main driver and the Phase-1 diagnosis
  ("ceiling from data/preprocessing", master §13.4) is wrong.

**My expectation: `2d` lands well above 0.62 — likely 0.72–0.78.** Reasoning: the Phase-1 models were *also*
2D and plateaued at 0.62 regardless of capacity (31M → 102M params, master §13.4), which points at the
pipeline rather than dimensionality. nnU-Net's `2d` shares all of nnU-Net's preprocessing (resampling to
common spacing, CTNormalization, augmentation) and the clean both-annotated labels, differing from Phase-1
almost only in those. So `2d` should recover most of the gap, and the `3d_fullres − 2d` difference is the
clean measurement of what 3D itself is worth. A useful related data point already exists: E5 showed
ensembling contributes ~nothing, so the gain must sit in {3D, preprocessing, labels} — E1 splits the first
from the other two.

## Note for the synthesis (S1)

E1 and E2 together are what make the attribution quantitative. E5 (done) has already removed ensembling from
the four confounded axes. E1 measures the 3D axis; E2 measures the label axis; preprocessing is then the
remainder. Do not write S1 until both tracks are ✅ (per `AGENT_INSTRUCTIONS.md` §7).
