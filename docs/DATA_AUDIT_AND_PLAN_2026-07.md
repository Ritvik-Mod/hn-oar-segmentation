# Data Audit + Deliverable-Model Plan — July 2026

> Written after pivoting from "research paper" to "portfolio-grade working segmentation model" for internship applications. Compute: HPC (rachel, 2×L40) usable; full dataset now local on Mac. Timeline ~1 month.

## 1. Data audit (150-patient random sample of ML_Dataset_Final)

Annotation is **partial and inconsistent** — clinicians contour only what is clinically relevant per patient, so a missing mask does NOT mean the organ is absent. This is the single most important modeling constraint.

Reliably-annotated head-and-neck OARs (share of patients with ≥1 positive slice):

| Structure | % patients | Median +px/slice | Modeling note |
|---|---|---|---|
| EYE_L / EYE_R | 65% / 64% | ~235 | good bilateral target |
| OPTIC_NERVE_L / _R | 65% / 15% | ~60 | small, R under-annotated |
| TEMPORAL_LOBE | 61% | large | easy, optional |
| SPINAL_CORD_PRV / SPINAL_CORD | 54% / 41% | ~135–320 | key OAR |
| LARYNX | 51% | ~544 | good target |
| PAROTID_L / PAROTID_R | 46% / 45% | ~310–350 | primary clinical target |
| OPTIC_CHIASMA | 42% | ~260 | tiny, hard |
| LENS_R / LENS_L | 59% / 15% | ~20 | very small, hardest |
| BODY | 35% | ~38k | trivial outline |
| PTV (tumour) | 92% | ~2600 | target volume, not an OAR — handle separately |
| BRAINSTEM | 5% | large | too rare to model reliably |

Thorax/abdomen structures (LUNG, LIVER, BREAST, FEMUR, KIDNEY, HEART…) appear only because non-H&N patients are mixed in — exclude from the H&N model.

**Also found:** npz mask KEYS still contain many non-standardized raw variants (`dx22`, `DLX1`, `X2`, `spill`, …). Label standardization logged actions but did not rewrite all keys — a cleaning/canonicalization pass is required before multi-organ training.

**Verdict:** multi-organ is feasible. Curated target set (~8): parotids L/R, eyes L/R, spinal cord, larynx, optic nerves L/R (+ optic chiasm / temporal lobe optional).

## 2. Solutions to the current problems

- **Dataset heterogeneity (spacing 0.64–0.98mm, scanner/protocol, two parse pipelines):** solve by resampling to a common spacing + intensity normalization + strong augmentation. nnU-Net does all of this automatically. This directly addresses the "dataset variation issue."
- **Dice plateau at ~0.62:** the plateau is a symptom of from-scratch training + 2D slice modeling + no resampling, not architecture. nnU-Net (proper preprocessing, deep supervision, 3D/2.5D, 5-fold ensemble) typically reaches ~0.80+ Dice on parotids.
- **Partial labels:** use a masked / marginal loss so un-annotated organs are excluded from the loss, not treated as background. For nnU-Net multi-organ, train per-organ or region-grouped models on cases where those organs are annotated.
- **Mislabeled masks (12 parotid patients, oversized masks):** keep the exclusion; add an automated QC filter (area/eccentricity thresholds) as a reusable cleaning step.
- **Test set never evaluated:** run the locked 165-patient test set ONCE at the end, on the final model only.

### Friends' suggestions — evaluated
- **Mixture of Experts:** rejected for the core model. Heterogeneity is better handled by resampling/normalization/augmentation; MoE adds routing instability without a payoff at this scale. nnU-Net's built-in 5-fold ensemble captures the useful "multiple experts" benefit.
- **Split image into 4 sub-images:** rejected as naive quadrant tiling (bisects organs, kills global context). The principled equivalent is nnU-Net sliding-window patching or ROI cropping around the organ.

## 3. Staged build plan (~4 weeks)

- **Phase 0 — foundations (days 1–3):** clean repo scaffold; reusable data-audit + label-cleaning + QC scripts; NIfTI conversion for nnU-Net; fix stale hardcoded paths; config-driven, partial-label-aware dataloader.
- **Phase 1 — the guaranteed win (week 1–2):** nnU-Net v2 parotid model trained on HPC; evaluate on val AND the locked test set; big Dice jump documented; Gradio inference demo deployed to HuggingFace Spaces (clickable link for recruiters).
- **Phase 2 — multi-organ (week 3):** extend to curated ~8-OAR set with masked/partial-label handling; fold into the same demo.
- **Phase 3 — polish (week 4):** README with results table + prediction visuals; reproducibility; ONNX/TorchScript export + minimal FastAPI (deployable artifact); keep from-scratch models documented as architecture evidence.
