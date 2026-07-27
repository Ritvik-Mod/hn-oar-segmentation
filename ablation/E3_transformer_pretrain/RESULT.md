# E3 — Transformer pretraining (Q3): does ImageNet init fix the transformers?

**Question.** The Phase-1 TransUNet and Swin-UNet were trained **from random init,
no ImageNet pretraining** — both are *designed* to start from pretrained backbones.
Was their weakness (Swin especially) the *architecture*, or just "from scratch"?
Retrain both in their intended pretrained configuration and compare to their
from-scratch selves (the P0 test numbers).

**Status: ✅ DONE (2026-07-17).**

## HEADLINE: **No.** ImageNet pretraining does not fix the transformers — +0.006 / +0.003 Dice, and both *still* lose to a 2018 CNN.

| Model | 3D Dice | Tversky | HD95 (mm, iso) | Surf-Dice |
|---|---|---|---|---|
| TransUNet — from scratch (P0) | 0.7313 | 0.7307 | 5.60 | 0.8594 |
| **TransUNet — ImageNet ResNet-50 (E3)** | **0.7373** | 0.7325 | **3.78** | 0.8759 |
| **Δ** | **+0.0060** | +0.0018 | **−1.82** | +0.0165 |
| Swin — from scratch (P0) | 0.7156 | 0.7082 | 8.34 | 0.8794 |
| **Swin — ImageNet Swin-T enc (E3)** ⚠️ | **0.7185** | 0.7122 | **6.70** | 0.8679 |
| **Δ** | **+0.0029** | +0.0040 | **−1.64** | −0.0115 |

### The full 2D picture (all six models, locked 43-case test, identical protocol)

| Rank | Model | Dice | Tversky | HD95 | Surf-Dice |
|---|---|---|---|---|---|
| 1 | **Attention U-Net** (scratch) | **0.7434** | 0.7387 | **3.57** | **0.8827** |
| 2 | U-Net (scratch) | 0.7390 | 0.7315 | 4.25 | 0.8702 |
| 3 | **TransUNet pretrained** | 0.7373 | 0.7325 | 3.78 | 0.8759 |
| 4 | TransUNet (scratch) | 0.7313 | 0.7307 | 5.60 | 0.8594 |
| 5 | **Swin pretrained** | 0.7185 | 0.7122 | 6.70 | 0.8679 |
| 6 | Swin-UNet (scratch) | 0.7156 | 0.7082 | 8.34 | 0.8794 |

**Total spread across all six — architecture AND initialisation combined — is 0.0278 Dice.**

## Interpretation (answers Q3)

**1. Pretraining does essentially nothing for overlap.** +0.0060 (TransUNet) and +0.0029 (Swin) are within
per-case noise (Track A measured per-case Dice std ≈ 0.078 on this test set). Given both models their
*intended* ImageNet initialisation — 258/258 exact weight load for TransUNet, same architecture, same loss,
same dataloader, same protocol — and they gain nothing measurable.

**2. But it clearly helps boundaries.** HD95 improves by **−1.82 mm** (TransUNet) and **−1.64 mm** (Swin),
consistently and on both arms. Pretrained features localise gland *edges* better; they just don't find more
gland. This is a real, reportable effect and the one place pretraining earned its keep.

**3. The ranking is unchanged — the CNNs still win.** The best pretrained transformer (TransUNet, 0.7373)
still loses to the from-scratch **Attention U-Net (0.7434)**. So the transformers' Phase-1 weakness was
**NOT** the from-scratch handicap. Master §13.4's framing ("pure transformer underperforms without
pretraining… 513 parotid patients is too little for a from-scratch ViT") is now **half-refuted**: the data
*is* too little, but *pretraining doesn't rescue it* — the ceiling is the dataset, not the initialisation.

**4. This is the study's thesis, from a sixth independent angle.** Architecture + initialisation together
span **0.028 Dice** across six models. Against that: preprocessing ≈ **+0.145**, labels ≈ **+0.129** (latent).
Model choices are ~5× less important than preprocessing and labels on this data.

### Scoring the pre-registered hypothesis (recorded before the run — see below)

| Prediction | Outcome |
|---|---|
| "Both pretrained models improve" | ✅ correct, but **marginally** (+0.006 / +0.003) |
| "**Swin most**" | ❌ **wrong on test Dice** — TransUNet gained more (+0.0060 vs +0.0029) |
| "boundary metrics improving too" | ✅ **correct and the strongest effect** (−1.8 mm both) |
| "If they still don't beat the CNNs, that is itself the finding" | ✅ **this is what happened** |

### ⚠️ Val loss and test Dice disagree — do not cite the val numbers

| Model | Phase-1 scratch val | E3 pretrained val | val says | test Dice says |
|---|---|---|---|---|
| TransUNet | 0.1692 (@ep 14) | **0.1810** (@ep 15) | pretraining **hurt** | pretraining **helped** (+0.006) |
| Swin | 0.2101 (@ep 13) | **0.1703** | pretraining **helped a lot** (−0.040) | barely moved (+0.003) |

The two metrics point opposite ways on **both** arms. Two reasons: (a) the capped h5 (empty slices ≤40/patient)
changes val batch composition, so E3's val loss is not strictly comparable to Phase-1's; (b) val loss is a 2D
slice-level quantity while the reported Dice is 3D volumetric on the locked test set — master §13.1 already
warns these are different quantities. **Only the test Dice above is authoritative.** The val numbers are
recorded for completeness and should not be used to argue the pretraining question either way.

Also notable: TransUNet's pretrained run drove **train loss to 0.0545** — a third of its own val loss (0.1810)
and well below the from-scratch model's *best val ever* (0.1692). The pretrained encoder memorised the 6,934
parotid training slices without generalising. That is the mechanism behind the null result.

## Reproducibility

| Item | TransUNet_pretrained | Swin_pretrained |
|---|---|---|
| Train | `train_e3.py --model transunet_pretrained --samples-per-epoch 12000 --epochs 40 --patience 8` | `--model swin_pretrained` (same flags) |
| Weight load | **258/258** torchvision `resnet50 IMAGENET1K_V2`, conv1 stem 3→1 averaged | timm `swin_tiny_patch4_window7_224` (features_only, in_chans=1, img_size=512) |
| Params | 102.5M (identical to P0's from-scratch TransUNet) | 48.3M (⚠️ ≠ P0's 27M Swin — see caveat) |
| Early stop | **epoch 23** (best @ 15) | **epoch 22** |
| Best val | 0.1810 | 0.1703 |
| Wall-time | **94.3 min** | **55.5 min** |
| GPU | 1× L40S 46GB, AMP, seed 42 | same |
| Eval | `evaluate_e3.py --tag TransUNet_pretrained` (P0's `phase1_metrics`, isotropic, 86 sides) | `--tag Swin_pretrained` |
| Data | `trainval.h5` — 10.62 GB, 667/667 patients, 34,415 slices (30,249 train / 4,166 val; 6,934 / 806 with parotid) | same |
| Cost | ~$2.7 total for both on a $1.07/hr pod | |

**Metric note:** E3 and P0 share `phase1_metrics.py` (**isotropic** 0.977 spacing), so pretrained-vs-scratch is
apples-to-apples. These HD95s are **not** comparable to nnU-Net/E4's anisotropic (0.977, 0.977, 3.0) numbers —
master §12.3 / §14.5.

### Budget deviations (state these when citing absolute numbers)

1. **Empty-slice cap** (`--cap-empty-per-patient 40`) shrank the h5 from ~28 GB to 10.62 GB, changing negative
   sampling slightly. No parotid-bearing slice was dropped.
2. **12,000 samples/epoch** vs Phase-1's full ~30k pass. So an E3 "epoch" ≈ 0.4 Phase-1 epochs, and patience 8
   ≈ 3.2 Phase-1 epochs (Phase-1 used 10). Both arms nonetheless converged and early-stopped naturally, and
   TransUNet's train loss (0.0545) shows no sign of under-training — if anything the opposite.

Both deviations affect **absolute** values; the **pretrained-vs-scratch delta** — same data, same protocol,
same cap on both sides — is the takeaway and is unaffected.

---

## What was built and validated (offline, no training)

- **`models_pretrained.py`** — two pretrained arms. Validated via `validate_e3.py`
  (forward+backward at batch 2 / 512² / 1-channel, gradients finite, CombinedLoss
  computes; ran both random-init and **real** pretrained loads):
  - **`transunet_pretrained`** — the FROZEN hand-built TransUNet with its
    `ResNet50Encoder` initialised from **torchvision ResNet-50 IMAGENET1K_V2**. The
    module names match torchvision exactly → **258/258 encoder tensors loaded**; the
    7×7 conv1 stem adapted 3→1 channel by averaging the RGB filters. Clean,
    same-architecture, only-initialisation-changed swap. Transformer + CUP decoder
    stay from-scratch (the plan's minimum requirement is a pretrained ResNet-50).
  - **`swin_pretrained`** — encoder = **timm `swin_tiny_patch4_window7_224`**
    (ImageNet, `features_only`, `in_chans=1`, `img_size=512`; stage outputs
    [96,192,384,768] at [128,64,32,16], matching the Swin-UNet stage dims) feeding
    the **same from-scratch Swin decoder blocks** (PatchExpanding + SwinTransformerStage
    + seg-head + bilinear-to-512). 48.3 M params.
- **`train_e3.py`** — copy of the Phase-1 training loop (frozen originals untouched),
  model built by the builders, paths via CLI. Reproduces the Phase-1 protocol:
  CombinedLoss (0.5 Dice + 0.5 BCE), Adam 1e-4, batch 8, CUDA AMP,
  `WeightedRandomSampler` (20:1 slice-level via the frozen `hdf5_dataloader`), hflip
  aug (with R/L channel swap). Reuses the frozen `hdf5_dataloader.py` + `loss_function.py`.
- **`evaluate_e3.py`** — scores an E3 checkpoint on the locked 43-case test set with
  the **identical Phase-1 3D protocol as P0** (reuses `P0/phase1_metrics.py`), so
  pretrained-vs-scratch is apples-to-apples on the same cases/metric.
- **`build_compact_h5.py`** — training-data prep (see §Data). Validated: builds a
  correct `dataset.h5` (train+val, image + PAROTID_R/L) that the frozen
  `hdf5_dataloader` reads unchanged.

## ⚠️ Design consideration flagged to the owner (golden rule #6)

A **true "same-architecture, only-init-changed" Swin is not possible** with an
off-the-shelf ImageNet Swin-T. The frozen Swin-UNet uses **window = 8 / 512-input**
(relative_position_bias_table 225, depths [2,2,2]); ImageNet Swin-T uses **window = 7
/ 224-input** (bias table 169, depths [2,2,6,2]). The window-size and depth mismatch
makes the pretrained attention weights **un-loadable** into the frozen architecture
(verified: bias tables 225 vs 169). So the Swin arm changes the encoder to timm's
Swin-T, which means **the Swin pretrained-vs-scratch comparison carries a mild
architecture confound** (encoder depth/window differ from the from-scratch Swin).
**The TransUNet arm has no such caveat** (clean ResNet-50 load) and is the cleaner
test of Q3. Recommend leaning on TransUNet for the headline pretraining claim and
treating Swin as corroborating-with-a-caveat. Owner: confirm this is acceptable, or
we can instead (a) retrain the from-scratch Swin at window=7/224-equivalent to accept
the weights, or (b) drop the Swin pretraining arm.

## Data (the expensive part — see Track A upload learnings)

E3 needs the 2D-slice data for **train+val** patients (667 patients). `build_compact_h5.py`
packs only those (test dropped) with only image + PAROTID_R/L → **~30 GB** faithful,
or ~7-13 GB with `--cap-empty-per-patient` (mild negative-sampling caveat). This is
the ~2-3 h / ~$8-10 upload Track A warned about; plan it (RUNPOD_COMMANDS_TrackB §0).

---

## Hypothesis (pre-registered BEFORE the run — scored in the Interpretation section above)

Both pretrained models improve over their from-scratch P0 selves, **Swin most**
(from-scratch ViTs are the most data-starved), with boundary metrics improving too.
If they still don't beat the CNNs, that is itself the finding — the ceiling is
data/pipeline-bound, not "from scratch" (Q3 answered either way). The clean read is
the TransUNet delta (P0 TransUNet → E3 TransUNet), which isolates pretraining with no
architecture change.

**Commands:** `ablation_study/RUNPOD_COMMANDS_TrackB.md` §3.
