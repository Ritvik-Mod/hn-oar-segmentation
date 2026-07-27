# E8 — HIGH-QUALITY TEST-SET PREDICTION GALLERY (rendering only, $0, no GPU)

> **One Claude Code agent.** Render **high-quality visualisations of the test-set predictions** for all the
> important models in the study (E1–E7 + the nnU-Net references + the from-scratch baselines), overlaying each
> model's prediction on the CT with the ground-truth contour, plus per-case 3D gland renders and
> **same-case-across-models comparison grids**. **Everything already exists as prediction files — this is
> pure rendering, no training, no inference, no pod, no cost.** Read `AGENT_INSTRUCTIONS.md` + the READ-FIRST
> list before starting. Owner: Ritvik Mod.

---

## 0. READ FIRST (context — in this order)
1. **`MASTER_PROJECT_REFERENCE.md`** — the project brain. Focus §3 (dataset, the parotids, R=label 1/L=label 2,
   HU/spacing), §14 (nnU-Net pipeline), §20 (the full ablation + §20.0 results table, §20.E E6, §20.F E7),
   §22 (artifact map). This tells you what every model IS and how they compare.
2. **`PROJECT_NARRATIVE.md`** — canonical numbers/framing (never quote "0.62 → 0.82"; honest = 0.743 → 0.819).
3. **`ablation_study/SYNTHESIS.md`** — the findings, so your captions/labels are scientifically correct.
4. **`pipeline/test_and_visualize.py`** — the EXISTING renderer (2D axial montages: CT + pred R=red/L=blue +
   GT dashed, and 3D marching-cubes gland renders + an index.html gallery). **Reuse/extend this** — it already
   handles nnU-Net-style `.nii.gz` labelmaps (1=parotid_r, 2=parotid_l). Convention: **R = red `#ff3b30`,
   L = blue `#0a84ff`, GT = dashed contour in the matching colour**; soft-tissue window (hu_offset 8192, WL 40,
   WW 400).
5. **`ablation_study/E7_singleside/`** (build_singleside.py, eval_singleside.py) and the E6/E7 `RESULT.md` —
   for the single-side test set and how models were scored there.

## 1. WHERE EVERYTHING LIVES (exact paths — do not hunt)

**Both-parotid test set (43 cases):** CT = `Parotid-Project/Datasets/Dataset002_Parotid/imagesTs/PARxxxx_0000.nii.gz`,
GT = `Parotid-Project/Datasets/Dataset002_Parotid/labelsTs/PARxxxx.nii.gz` (label 1=R, 2=L). Predictions (all
`.nii.gz` labelmaps in this same geometry — a **clean apples-to-apples set**):

| Model | Predictions dir |
|---|---|
| nnU-Net clean-208, single fold (0.8187) | `Parotid-Project/Results/pred_nomirror` (or `test_predictions`) |
| nnU-Net 5-fold ensemble (0.8202, BEST) | `Parotid-Project/Results/pred_ensemble` |
| E1 — nnU-Net 2d | `ablation_study/_pod_results/preds/E1_2d` |
| E2 — nnU-Net dirty-labels 430 | `ablation_study/_pod_results/preds/E2_dirty` |
| E2b — nnU-Net gapped-208 | `ablation_study/_pod_results/preds/E2b_gapped` |
| E4 — hand-built 3D U-Net (raw + CC) | `ablation_study/_pod_results_B/E4_pred_test` , `..._B/E4_pred_test_cc` |
| E6 — masked-430 (raw + CC) | `ablation_study/_pod_results_B/E6_masked_loss/pred_masked430` , `..._cc` |
| E6 — dirty-430 (raw + CC) | `ablation_study/_pod_results_B/E6_masked_loss/pred_dirty430` , `..._cc` |
| **E7 — per-side experts** (combined + each) | `ablation_study/E7_experts/_pod_results/preds/{experts_both,left_both,right_both}` |

**Single-side test set (58 cases):** CT + GT are built by `ablation_study/E7_singleside/build_singleside.py`
(re-run it if the volumes aren't cached; it's CPU/instant). Predictions:
`ablation_study/E7_experts/_pod_results/preds/{experts_ss,left_ss,right_ss}` and the other models scored on
single-side are under the same `preds/` parent (`nnunet_clean208`, `e4`, `masked430`, `dirty430`).

**From-scratch 2D baselines (P0: U-Net / Attention / TransUNet / Swin):**
`ablation_study/_pod_results_B/P0_preds` — ⚠️ **these are `.npz`** (keys `pred_r`, `pred_l`, `gt_r`, `gt_l`)
in the **original 512×512 slice geometry**, NOT nnU-Net-resampled. Render them overlaid on the **original
per-slice CT** (reconstruct via `pipeline/build_volumes.py`, or the npz `image`), NOT on the Dataset002
imagesTs (different spacing). Keep them as a **secondary** panel; the `.nii.gz` models above are the primary
comparison set.

Per-case scores to pick representative cases: `Parotid-Project/Visualizations/parotid_viz_testset/metrics.csv`
(nnU-Net single per-case), `..._ensemble/metrics.csv`, and `ablation_study/E7_experts/_pod_results/results/
eval_singleside_percase.csv`. Or just compute per-case Dice on the fly (pred vs GT — cheap).

## 2. WHAT TO RENDER (high quality)

For **each model** (both-parotid set primary; single-side set for the models scored there):
1. **Per-case 2D axial montage** — CT (soft-tissue window) + predicted **R red / L blue** (semi-transparent
   fill AND crisp contour) + **GT dashed contour** in the matching colour, ~8–12 gland-bearing slices evenly
   spread, per-case **Dice in the title**. **High quality: dpi ≥ 200**, clean layout, legend.
3. **Per-case 3D surface render** — marching-cubes of the predicted glands (R red / L blue), a couple of
   viewing angles, cropped to the glands. (Reuse `test_and_visualize.py --render3d`.)
3. **Same-case-across-models comparison grids** — THE most useful output. For a set of **representative cases**
   (pick from per-case Dice: best, median, worst, plus 1–2 single-side cases and, if any, an L/R-confusion
   case), render **one figure per case showing the SAME axial slice with every model's prediction side by
   side** (small multiples), GT dashed in each, model name + that model's Dice under each panel. This lets the
   owner eyeball where models differ (stray islands for Swin, boundary tightness, L/R behaviour, single-side
   contralateral prediction).
4. **An `index.html` gallery** (reuse the tool's writer) linking every case, grouped by model, plus a
   dedicated "comparison grids" section.

**Models to include** (the important/recent ones + baselines, as requested): nnU-Net single fold, nnU-Net
5-fold ensemble, E1 2d, E2 dirty, E2b gapped, E4 (+CC), E6 masked-430 (+CC), E6 dirty-430 (+CC), E7 experts
(combined + left-only + right-only), and the P0 baselines (U-Net, Attention U-Net, TransUNet, Swin-UNet).
Prefer the **+CC (post-processed)** version where a model has one, and note it in the caption.

## 3. OUTPUT
- Everything under **`ablation_study/prediction_gallery/`**:
  `prediction_gallery/<model>/PARxxxx_slices.png`, `PARxxxx_3d.png`, and
  `prediction_gallery/comparison/case_<PARxxxx>_allmodels.png`, plus `prediction_gallery/index.html`.
- A short **`prediction_gallery/README.md`**: which models, which cases, the colour/label convention, the CC
  note, and the honest per-model Dice (from §20.0) so the pictures are captioned with correct numbers.
- Copy nothing out of read-only zones; only READ predictions/CT/GT. New files only under `prediction_gallery/`.

## 4. NOTES / GOTCHAS
- **No GPU, no pod, no training** — pure `nibabel` + `numpy` + `matplotlib` (+ `scikit-image` for 3D) locally.
- Labels are **1 = parotid_r, 2 = parotid_l** in the nii.gz; keep R=red / L=blue consistent everywhere.
- The `.nii.gz` models share one geometry → their comparison grids are exact. The P0 `.npz` baselines are in
  original slice geometry → render them separately (don't force them into the resampled grid).
- Use the same window/level and the same slice indices within a comparison grid so panels are comparable.
- If `build_singleside.py` output isn't cached, re-run it (CPU, instant) to get the single-side CT/GT.

## 5. DOC-UPDATING (light — this is a viz deliverable, not an experiment)
- Write `prediction_gallery/README.md` (above). Append a one-paragraph note under **`MASTER_PROJECT_REFERENCE.md`
  §20.F** (or a new §20.G "prediction gallery") pointing to the folder — append-only, don't edit §1–19 or the
  frozen ablation logs. No SYNTHESIS change needed.

## 6. HANDOFF PROTOCOL (standing instruction)
When the owner types **"rdy for handoff"**, STOP and emit one self-contained handoff prompt (copy-paste block):
what's rendered so far (which models/cases done, output paths), what's left, any gotchas (geometry, missing
preds), and instruct the owner to paste Appendix A + that state into a fresh agent. Warn the owner if you sense
you're low on context.

---

## APPENDIX A — NEW-AGENT KICKOFF PROMPT (owner pastes this to start a fresh agent)

```
You are a Claude Code agent working in the "ct scan project" folder (head-and-neck CT parotid-gland
segmentation). Your job: render HIGH-QUALITY test-set prediction visualisations for the study's models. This
is PURE RENDERING from prediction files that already exist — no training, no inference, no GPU, no pod, $0.

STEP 1 — GET CONTEXT (read first, in order):
  1. MASTER_PROJECT_REFERENCE.md — project brain. §3 (dataset; R=label 1, L=label 2; HU/window), §14 (nnU-Net),
     §20 (ablation + §20.0 results table with every model's Dice, §20.E E6, §20.F E7), §22 (artifact map).
  2. PROJECT_NARRATIVE.md — canonical numbers (never "0.62 → 0.82"; honest = 0.743 → 0.819).
  3. ablation_study/SYNTHESIS.md — findings, so captions are correct.
  4. pipeline/test_and_visualize.py — the EXISTING renderer to reuse/extend (CT + pred R=red/L=blue + GT dashed
     montages, 3D gland renders, index.html). Convention: R red #ff3b30, L blue #0a84ff, GT dashed.
  5. ablation_study/E8_prediction_gallery_tasks.md — YOUR FULL TASK: exact prediction/CT/GT paths for every
     model (both-parotid 43 + single-side 58 + the .npz P0 baselines), what to render (per-case montages, 3D,
     and same-case-across-models comparison grids), output layout, and gotchas (nii.gz vs npz geometry).
  6. ablation_study/AGENT_INSTRUCTIONS.md — golden rules (read-only zones; new outputs only under the gallery
     folder; never modify checkpoints/data/frozen docs/webapp).

STEP 2 — HOW TO WORK:
  - Everything is local: nibabel + numpy + matplotlib (+ scikit-image for 3D). No pod, no GPU, no cost.
  - Predictions already exist (paths in the task doc §1). Only READ them + the CT/GT; write only under
    ablation_study/prediction_gallery/.
  - Render per §2 at dpi ≥ 200. The .nii.gz models (nnU-Net/E1/E2/E4/E6/E7) share one geometry → exact
    comparison grids; the P0 .npz baselines are original 512×512 → render separately.
  - Caption every figure with the correct per-model Dice from MASTER §20.0. Keep R=red / L=blue everywhere.

STEP 3 — EXECUTE per the task doc; write prediction_gallery/README.md + an index.html; append a pointer under
MASTER_PROJECT_REFERENCE.md §20 (append-only). When I type "rdy for handoff", follow the handoff protocol in
the task doc §6.

[PASTE PRIOR-SESSION STATE HERE — or "fresh start, nothing done yet" if this is the first agent.]
```
