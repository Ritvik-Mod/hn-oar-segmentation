# TRACK B — custom-PyTorch experiments (Claude Code Account #2)

> You are **Track B**. You own **P0, E3, E4, A1**. All of these use the from-scratch codebase and/or existing
> checkpoints; none of them need anything from Track A (nnU-Net). First read `AGENT_INSTRUCTIONS.md`, then
> `ABLATION_PLAN.md`, with `MASTER_PROJECT_REFERENCE.md` as context. Follow the golden rules (locked test
> set, read-only zones, copy-don't-edit the frozen originals).
>
> Output under `ablation_study/P0_phase1_on_test/`, `E3_transformer_pretrain/`, `E4_custom_3d_unet/`,
> `A1_swin_failure/`. Keep the status table + master §20.B live as you go.

## What you have to start from
- The four from-scratch models + training/eval code (master §11–§13): `unet.py`, `attention_unet.py`,
  `transunet.py`, `swin_unet.py`, their `*_train.py` / `*_evaluate.py`, the `hdf5_dataloader.py` /
  `pytorch_dataloader.py`, and `loss_function.py`.
- The four existing checkpoints in `checkpoints/` (U-Net, Attention, TransUNet, Swin) — **read-only**.
- The 3D evaluator `pipeline/eval_testset.py` and the Phase-1 3D volumetric eval logic inside the
  `*_evaluate.py` scripts.
- The locked test patients (Dataset002 test = 43 both-parotid cases) and their labels.
- Data access: the loose `.npz` / `dataset.h5` (external drive or HPC) — needed for E3/E4/P0 training.

## Suggested order: **P0 → A1 (both cheap, CPU) → E3 → E4.**

---

## P0 — Re-evaluate the 4 Phase-1 checkpoints on the TEST set  *(makes the whole study comparable)*
The four from-scratch models were only ever scored on the **validation** set. Score all four on the **locked
test set** (the same 43 both-parotid cases) using the Phase-1 3D volumetric metric (stack predicted slices
per patient → Dice / Tversky / HD95 / Surface-Dice). Reuse the `*_evaluate.py` logic but point it at the
test patients.
- **Output:** `ablation_study/P0_phase1_on_test/eval.csv` with one row per model + `RESULT.md`.
- **Why:** gives the study a single comparable "test" column and lets the conference abstract report Phase-1
  on test (fixes the current val-vs-test caveat). CPU is fine.

## A1 — Why Swin-UNet's boundary metrics are bad  *(analysis only, NO training, Q5)*
Confirm the two hypotheses from the master doc (§11.4, §13) using the **existing** Swin checkpoint/predictions:
1. **Coarse-output hypothesis:** Swin's final prediction is produced at **128×128 then bilinearly upsampled
   to 512×512** (memory workaround) → inherently blurry boundaries → inflated HD95 / low Surface-Dice.
2. **Underfit/stray-island hypothesis:** Swin floored at val loss 0.21 (vs ~0.16 CNNs) → scattered
   false-positive voxels far from the gland → HD95 (95th-pctile distance) blows up.
- **Do:** (a) count connected components per predicted gland for Swin vs a CNN (expect many more stray
  islands for Swin); (b) measure distance of stray components from the main gland; (c) render a side-by-side
  montage (Swin coarse/island-y prediction vs U-Net crisp prediction on the same slice).
- **Output:** `ablation_study/A1_swin_failure/RESULT.md` + figures. No GPU needed.

## E3 — Transformer pretraining  *(isolates the from-scratch handicap, Q3)*
Retrain **TransUNet** and **Swin-UNet** in their *intended* pretrained configuration and compare to their
from-scratch selves (use the P0 test numbers as the from-scratch baseline).
- **TransUNet:** load an **ImageNet-pretrained ResNet-50** into the encoder (and, if feasible, an
  ImageNet/ViT-pretrained transformer; at minimum pretrain the ResNet-50 backbone). Adapt the 3-channel stem
  to 1-channel CT (average or repeat the pretrained conv1 weights).
- **Swin-UNet:** load **ImageNet-pretrained Swin-T** weights into the encoder; adapt the patch-embed stem to
  1-channel.
- Keep the Phase-1 protocol otherwise identical (combined Dice+BCE, Adam 1e-4, batch 8, AMP, same 2D
  dataloader + weighted sampling, same augmentation). Copy the training scripts into
  `ablation_study/E3_transformer_pretrain/` before modifying — **don't edit the frozen originals.**
- **Evaluate on the TEST set** with the Phase-1 3D volumetric eval (same as P0).
- **Compare:** pretrained TransUNet/Swin vs from-scratch TransUNet/Swin (P0 numbers).
- **Expected:** both improve, Swin most; boundary metrics improve. If they still don't beat the CNNs, that's
  itself a finding (data-bound ceiling dominates).
- **Output:** `ablation_study/E3_transformer_pretrain/` — checkpoints, `eval.csv`, `RESULT.md`.
- **Compute:** 2 × 2D transformer trainings; HPC if a GPU is free, else a cheap cloud GPU.

## E4 — Hand-built 3D U-Net vs nnU-Net  *(isolates architecture vs pipeline, Q4)*
Implement a **plain 3D U-Net** (extend `unet.py` to 3D: Conv3d/MaxPool3d/ConvTranspose3d, features e.g.
[32,64,128,256,320]) and train it on **nnU-Net-style preprocessed 3D volumes** of Dataset002 so the *only*
difference from nnU-Net is the code, not the pipeline:
- Preprocess: reconstruct 3D volumes (`pipeline/build_volumes.py`), resample to **[3.0, 0.977, 0.977]**,
  **CTNormalization**, patch-based training (patch ~[48,224,192] or a memory-feasible size), same
  train/val/test patients as Dataset002.
- Loss: combined Dice+BCE (or Dice+CE) with sigmoid; single fold; standard Adam schedule.
- **Evaluate on the test set** with `pipeline/eval_testset.py`.
- **Compare:** custom 3D U-Net vs nnU-Net `3d_fullres` single fold (0.8187) — same data, same preprocessing,
  different implementation.
- **Expected:** lands close to nnU-Net (~0.78–0.82) → **architecture is nearly irrelevant once the pipeline
  is right** (the study's strongest thesis).
- **Fallback if replicating nnU-Net preprocessing is too costly:** note it and flag that the cheap proxy is
  nnU-Net's residual-encoder preset (a Track-A nnU-Net task) — but prefer the true custom 3D U-Net for the
  cleaner claim.
- **Output:** `ablation_study/E4_custom_3d_unet/` — checkpoint, `eval.csv`, `RESULT.md`.
- **Compute:** the most involved run; HPC or cloud GPU with enough VRAM for 3D patches.

---

## When you finish
- P0/E3/E4/A1 each have a `RESULT.md`, filled status rows (P0/E3/E4/A1) in `ABLATION_PLAN.md`, and appended
  entries under **master §20.B**. Add a "Track B complete" line under §20.B.
- Do **not** run S1 unless the status table shows Track A (E1/E2/E5) also ✅. Whichever agent sees both
  tracks ✅ may write `ablation_study/SYNTHESIS.md`.
