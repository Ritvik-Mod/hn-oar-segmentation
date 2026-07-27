# E8 — Test-set prediction gallery

High-quality visualisations of the **test-set predictions** for every important model in the
ablation study. **Pure rendering** of prediction files that already exist — no training, no
inference, no GPU, $0. Open **`index.html`** to browse everything.

## What's here

| Output | Path |
|---|---|
| Click-through gallery (all models + comparison section) | `index.html` |
| **Best-of showcase** (each model's top cases + overall leaderboard) | `showcase/showcase.html` |
| Per-case axial montages | `<model>/PARxxxx_slices.png` |
| Per-case 3D surface renders | `<model>/PARxxxx_3d.png` |
| Same-case-across-models comparison grids | `comparison/case_PARxxxx_allmodels.png` |
| P0 baseline comparison grids | `comparison/case_PARxxxx_p0baselines.png` |
| Per-case Dice, all models | `metrics_percase.csv` |

**Best-of showcase** (`showcase/`): a curated, self-contained folder — each model's **top 8 cases by
per-case Dice** (montage + 3D copied in), plus a **hall of fame** (`showcase/_hall_of_fame/`) of the 15
highest-Dice full both-parotid predictions across the whole study. Open `showcase/showcase.html`.

**Slice selection (why montages look aligned):** montage panels are spread over the **ground-truth
z-extent** (`pick_slices(..., anchor=GT)`), so every panel is centred on a real gland contour. An
earlier version spread over the pred∪GT union, which put the first/last panels on one-sided
**gland-tip slices** (pred present one slice beyond GT, or vice versa) — these contribute ~nothing to
Dice but looked like gross misses. Verified there is **no misalignment**: pred/GT centroids agree to
<1 voxel and per-case Dice reproduces §20.0 exactly; the tips were purely a slice-choice artifact.

Render scripts (re-runnable, resumable): `galcommon.py` (shared config + renderers),
`compute_metrics.py`, `render_all.py`, `make_comparison.py`, `render_p0.py`, `build_index.py`.

## Conventions (kept everywhere)

- **Right parotid = red `#ff3b30`**, **Left parotid = blue `#0a84ff`** (labels 1=R, 2=L in the
  `.nii.gz`). The single-side E7 experts store their one gland as label 1; each is remapped to the
  correct anatomical side (left-expert → blue, right-expert → red).
- Prediction = semi-transparent fill **+ solid contour**; **ground truth = cream dashed contour**
  (`#f4efe3`) so it reads clearly against any coloured fill on the dark background.
- Soft-tissue window: hu_offset 8192, **WL 40 / WW 400**. Montages/grids at **dpi 200**.
- Montage titles carry the **per-case Dice** (and R/L split); the aggregate test Dice (from
  `MASTER_PROJECT_REFERENCE.md` §20.0) is shown as `[agg …]`.

**Rendering style** (adopted from the project's website renderer
`Parotid-Project/Website/_parotid-render-tools/render_cases.py` for presentation quality, while
keeping the analysis annotations this gallery needs): **black background**, each montage panel
**head-cropped** (largest tissue component, couch/table excluded) so the head fills the frame,
**bilinear** CT interpolation. 3D renders **gaussian-smooth** the mask (σ 0.9) before marching cubes
to kill the 3 mm z-staircase, then use a **shaded `plot_trisurf`** surface at a 3/4 anterior view.

## Dataset

Locked **43-case both-parotid test set** (PAR0209–PAR0252, PAR0240 QC-dropped). CT from
`Dataset002_Parotid/imagesTs`, GT from `.../labelsTs`. All primary models are `.nii.gz` labelmaps in
**one shared geometry** → the comparison grids are exact, apples-to-apples.

## Models & honest per-model Dice (locked test set, from §20.0)

**Primary (nnU-Net-style `.nii.gz`), best → worst:**

| Model | Agg 3D Dice | Note |
|---|---|---|
| nnU-Net 5-fold ensemble | **0.8202** | best overall; delta vs 1 fold n.s. (p=0.35) |
| nnU-Net 3d, 1 fold | 0.8187 | reference |
| E1 nnU-Net 2d | 0.8117 | 2D vs 3D → 3D worth only +0.007 |
| E7 per-side experts (combined) | 0.8099 | "decompose" arm; does **not** beat nnU-Net |
| E4 hand-built 3D U-Net **+CC** | 0.7750 | largest-CC postproc collapses HD95 to ≈nnU-Net |
| E2 nnU-Net dirty-430 | 0.7726 | annotation gap, realistic net −0.046 |
| E6 custom U-Net masked-430 **+CC** | 0.7589 | annotation-gap fix (masked loss); < clean-208 |
| E6 custom U-Net dirty-430 **+CC** | 0.7296 | penalise arm; diverged in training |
| E2b nnU-Net gapped-208 | 0.6899 | pure annotation-gap cost −0.129 |

Single-side experts on the both-parotid set (each scores only its one gland, per-case): **E7
right-only ≈ 0.824, E7 left-only ≈ 0.796** (these are the two halves that combine into the 0.8099
experts row).

**Secondary — P0 from-scratch 2D baselines** (`.npz`, keys `pred_r/pred_l/gt_r/gt_l`):

| Model | Agg 3D Dice |
|---|---|
| Attention U-Net | 0.7434 |
| U-Net | 0.7390 |
| TransUNet | 0.7313 |
| Swin-UNet | 0.7156 |

**Geometry note:** the P0 `.npz` were expected to be in a different (original 512×512) geometry, but
a direct check shows their `gt_r/gt_l` match `Dataset002/labelsTs` **exactly (Dice 1.000)** — same
512×512 in-plane, same 114 slices. So they are overlaid on the same Dataset002 CT and are directly
comparable; they are still shown in a **separate** section (their own comparison grids) as requested.

## The "+CC" note

E4 and both E6 arms have a **largest-connected-component post-processing** version (`…_cc`). This
gallery uses the **+CC** predictions (as instructed) because CC removes stray islands and collapses
HD95 to near-nnU-Net levels; the raw versions differ mainly by scattered false-positive components.

## Representative cases (for the comparison grids)

Picked by nnU-Net 1-fold per-case Dice: **best** PAR0228 (0.920), **median** PAR0222 (0.844),
**worst** PAR0223 (0.564), plus the largest cross-model **disagreement** case PAR0214 (spread 0.445 —
E6-masked and E2b badly under-segment the left gland while nnU-Net stays tight) and two quartile cases
(PAR0247, PAR0237).
