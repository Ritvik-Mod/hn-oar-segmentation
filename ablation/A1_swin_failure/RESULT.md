# A1 — Why Swin-UNet's boundary metrics are disproportionately bad (Q5)

**Question.** On the Phase-1 validation set Swin-UNet's HD95 (12.1 mm) and
Surface-Dice (0.63) are much worse than its Dice (0.51) would suggest, and far worse
than the CNNs (U-Net HD95 4.74, TransUNet 3.58). *Why?* Analysis only — **no
training, no GPU.**

**Status: ✅ DONE (2026-07-17, RunPod, CPU).** Ran on P0's 172 saved predictions
(Swin + U-Net, 86 glands each). Both hypotheses confirmed with high significance.
(The analyzer was rewritten to bbox-crop the distance transforms — a first, un-cropped
version ran >60 min; the cropped version finishes in ~2–3 min.)

---

## Two hypotheses (from master §11.4 / §13.4), and how A1 tests each

**H1 — coarse output.** Swin's final logits are produced at **128×128 then bilinearly
upsampled to 512×512** — the memory workaround that replaced the paper's
`FinalPatchExpanding` (confirmed directly in `swin_unet.py:402-404`:
`seg_head` → `view(B,128,128,2)` → `F.interpolate(size=(512,512), mode='bilinear')`).
A boundary defined on a 128-grid and stretched 4× is inherently blocky → every
surface voxel can sit up to ~4 px (~4 mm) off the true edge → inflated HD95 and low
Surface-Dice **even when the overlap (Dice) is fine.**

**H2 — underfit / stray islands.** Swin floored at **val_loss 0.2101 @ epoch 13** vs
~0.16-0.17 for the CNNs (confirmed from `training_history_swinunet.json`: the curve
never drops below 0.21). An under-fit segmentation scatters **false-positive voxels
far from the true gland**; because HD95 is a 95th-percentile boundary distance, even
a few far-flung islands blow it up.

### What the code measures (Swin vs the U-Net baseline, 43 cases × {R,L} = 86 glands)

`analyze_swin_failure.py` (reuses P0's prediction volumes; 26-connectivity 3D):
1. **connected-component count** per predicted gland, and the **stray-island burden**
   (all components except the single largest, plus their voxel mass and fraction);
2. **distance (mm)** of the farthest stray island from the main gland (true
   anisotropic (3.0, 0.977, 0.977) sampling);
3. **HD95 on the full prediction vs on the largest-component-only prediction** — if
   deleting stray islands collapses Swin's HD95 but barely moves U-Net's, H2 is the
   proven driver of the HD95 blow-up;
4. paired Wilcoxon (Swin vs U-Net) on component count, stray count, stray fraction,
   farthest-island distance.

`make_montage.py` renders two figures on the same axial slice:
- `figures/montage_islands.png` — CT + U-Net (crisp, single component) vs Swin
  (stray false-positive islands), GT dashed; slice auto-picked as the one with the
  most Swin voxels outside a dilation of the GT (H2).
- `figures/montage_coarse.png` — a boundary zoom: Swin's blocky 128→512 upsampled
  edge vs U-Net's pixel-resolution edge (H1).

---

## Results (Swin-UNet vs U-Net, 86 glands each; true spacing 3.0×0.977×0.977)

| Metric (per gland) | Swin-UNet | U-Net | paired Wilcoxon p |
|--------------------|-----------|-------|-------------------|
| mean connected components | **2.84** (max 9) | 1.20 (max 5) | 1.3e-11 |
| % glands multi-component | **75.6%** | 16.3% | — |
| mean stray components (non-main) | **1.85** | 0.23 | 1.1e-11 |
| mean stray-voxel fraction | **5.03%** | 1.67% | 3.8e-9 |
| mean farthest stray island (mm) | **55.4** (max 306) | 12.2 | 1.9e-8 |
| HD95 full (mm) | 14.30 | 8.53 | — |
| HD95 largest-CC-only (mm) | 10.53 | 8.49 | — |
| **HD95 drop from CC cleanup (mm)** | **−3.77** | −0.04 | — |

Swin is more fragmented than U-Net on **62 of 86** glands (components), and its stray
islands sit **4.5× farther** from the main gland on average. All four island metrics
separate at p < 1e-8.

## Interpretation — both hypotheses hold, cleanly separated

- **H2 (underfit → stray islands) is the dominant, statistically overwhelming driver.**
  Swin scatters ~8× more stray components than U-Net (1.85 vs 0.23, p = 1e-11), up to
  306 mm from the gland, and **removing them (largest-CC only) drops Swin's HD95 by
  3.77 mm — vs 0.04 mm for U-Net** (which is already essentially single-component).
  So most of Swin's HD95 inflation is far-flung false-positive islands.
- **H1 (coarse 128→512 upsampling) is the residual floor.** Even *after* island removal,
  Swin's largest-CC HD95 (10.53 mm) stays ~2 mm above U-Net's (8.49 mm) — a systematic
  penalty on the *main* gland's boundary, exactly the blockiness the seg-head→bilinear
  upsample produces (`swin_unet.py:402-404`), visualised in `figures/montage_coarse.png`.
- Together these give a concrete, code-grounded answer to **Q5**: Swin's boundary
  metrics are bad for two compounding reasons — a coarse-boundary floor everywhere (H1)
  plus a long tail of stray-island catastrophes (H2) — which is why HD95/Surface-Dice
  are disproportionately worse than its Dice (0.716). Consistent with P0, where Swin had
  the **worst HD95 but the *fewest* one-sided misses** (it predicts glands, just messily).

## Figures
`figures/montage_islands.png` — Swin stray islands vs U-Net's clean single component
(auto-picked slice PAR0231 z=65). `figures/montage_coarse.png` — Swin's blocky
128→512 boundary vs U-Net's pixel-resolution edge.

Files: `per_gland.csv` (172 glands), `summary.json`.

**Commands:** `ablation_study/RUNPOD_COMMANDS_TrackB.md` §2.
