# E5 — Ensembling delta: single fold vs 5-fold ensemble

**Track A · Status: ✅ done · Date: 2026-07-16**

---

## Question

How much does nnU-Net's 5-fold ensemble actually buy over a single fold (fold 0), holding everything else
fixed (Dataset002, `3d_fullres`, `nnUNetTrainer_250epochs_noMirror`, `--disable_tta`, same locked 43-case
test set)?

This is the "ensembling" axis of the four confounded axes in the 0.62 → 0.82 headline (plan §0).

## Method

**No training and no inference was run for E5.** Both models were already trained and already scored on the
locked test set during the main project (master §15.2), so re-running them would only burn GPU hours to
reproduce existing numbers — the plan and the track file both say to tabulate the existing delta.

What E5 *adds* over the two already-published aggregate numbers is a **paired per-case analysis**. The
headline delta is **+0.0015 Dice**, which is far too small to interpret from two aggregate means: with 43
cases, a delta that size could easily be noise. A paired test is the only way to say whether it is real.

Inputs (all read-only, none modified):

| File | Role |
|---|---|
| `Parotid-Project/Visualizations/parotid_viz_testset/metrics.csv` | per-case metrics, single fold (43 cases) |
| `Parotid-Project/Visualizations/parotid_viz_ensemble/metrics.csv` | per-case metrics, 5-fold ensemble (43 cases) |
| `Parotid-Project/Results/results_tracker_TEST.csv` | headline single-fold row |
| `Parotid-Project/Results/results_5fold_TEST.csv` | headline ensemble row |

**Validation of the inputs:** the per-case files are the *same 43 cases* in both conditions (verified by set
equality — a hard assert in the script), and their means reproduce the published headline numbers exactly
(0.8187 / 0.8202 Dice, 5.25 / 5.24 mm HD95). So the per-case files are consistent with the headline results
and are a sound basis for the paired test.

### Command

```bash
python3 ablation_study/E5_ensembling/analyze_ensembling.py
```

### Config / reproducibility

| Item | Value |
|---|---|
| Seed | n/a — no training, no stochastic step; the analysis is deterministic |
| GPU | none — CPU only (Apple M-series, local Mac) |
| Wall-time | < 1 s |
| Cost | $0 |
| Models compared | `nnUNetTrainer_250epochs_noMirror__nnUNetPlans__3d_fullres`, fold 0 vs folds 0–4 ensemble |
| Test set | `Dataset002_Parotid/labelsTs`, 43 both-parotid QC-clean cases (locked, unchanged) |
| Script | `ablation_study/E5_ensembling/analyze_ensembling.py` |
| Output | `ablation_study/E5_ensembling/paired_per_case.csv` (43 rows) |

## Results

### Headline (both already on the locked test set)

| Model | 3D Dice | Tversky | HD95 (mm) | Surface Dice (3mm) | Source |
|---|---|---|---|---|---|
| noMirror single fold (fold 0) | 0.8187 | 0.8166 | 5.25 | 0.8965 | `results_tracker_TEST.csv` |
| noMirror 5-fold ensemble | 0.8202 | 0.8174 | 5.24 | 0.8990 | `results_5fold_TEST.csv` |
| **Delta (ensemble − single)** | **+0.0015** | +0.0008 | **−0.01** | +0.0025 | — |

### Paired per-case analysis (n = 43, ensemble − single fold)

| Quantity | Dice | HD95 |
|---|---|---|
| Mean delta | **+0.0016** | **−0.02 mm** |
| Median delta | +0.0066 | — |
| Std of delta | 0.0172 | — |
| Range of delta | −0.0553 … +0.0378 | — |
| Ensemble better / worse / tied | **26 / 17 / 0** | — |
| Wilcoxon signed-rank (two-sided) | W=395.0, **p = 0.346** | W=380.5, **p = 0.264** |

*(Mean delta is +0.0016 here vs +0.0015 in the headline table purely from rounding of the published
aggregates; the per-case computation is the more precise of the two.)*

**Scale context:** per-case Dice std across the test set is **0.0783**. The mean ensembling delta is
**0.020 ×** that spread — i.e. the effect is ~50× smaller than the case-to-case variation the model already
shows.

**Largest per-case swings** (both directions, showing the delta is churn rather than uniform gain):

| Case | Single | Ensemble | Delta |
|---|---|---|---|
| PAR0238 | 0.6577 | 0.6024 | −0.0553 |
| PAR0215 | 0.7289 | 0.7667 | +0.0378 |
| PAR0246 | 0.8037 | 0.8344 | +0.0307 |
| PAR0250 | 0.7417 | 0.7141 | −0.0276 |
| PAR0226 | 0.7673 | 0.7934 | +0.0261 |

## Interpretation

**The 5-fold ensemble buys nothing measurable on this dataset.** The +0.0015 Dice gain is not statistically
distinguishable from zero (Wilcoxon p = 0.346), and the ensemble is actually *worse* on 17 of 43 cases — it
reshuffles per-case results rather than uniformly improving them. HD95 is the same story (p = 0.264). This
holds for the boundary metrics too, so it is not a "Dice is insensitive" artifact.

**Why this matters for the study's thesis:** ensembling is one of the four confounded axes bundled into the
0.62 → 0.82 headline, and this experiment removes it from contention — **it accounts for ~0.15% of the
~20-point gain, i.e. essentially none of it.** Whatever drives the improvement is among the other three axes
(dimensionality, preprocessing, label quality). That is a useful negative result: it narrows the attribution
before E1/E2 even run.

**Practical consequence:** the single fold is the better deliverable. The ensemble costs 5× the training
compute and 5× the inference compute for a gain indistinguishable from noise. The main project already ships
the ensemble as "final" (master §15.2) — on this evidence that choice buys nothing, and for the portfolio/demo
the single fold is the defensible default.

**Caveat (worth stating rather than hiding):** n = 43 is small, so this is evidence of *"no detectable
effect at this sample size"*, not proof of exactly zero. But the direction is clear — any true effect is
comfortably smaller than the per-case noise, so it is not what explains the headline jump. Note also that the
folds share the same 208-case training pool and identical configuration, so the five models are highly
correlated; ensembling correlated models is expected to yield little, which is consistent with what we see.
