# E7 Part 2 — Per-side specialist models ("side-decomposition"), DESIGN CRITIQUE

> Not "mixture of experts": there is no gating network. These are **per-side specialist models** /
> a **side-decomposition ensemble**. Unrelated to the dataset-routing MoE rejected in master §14.

**Status: ✅ DONE (2026-07-18).** Design critique below (the "think first" deliverable); **execution + results
appended in §7–§9.** Chose **Option A** (two separate single-class nnU-Net experts); trained both on 2× A40.

---

## 0. Verified data pools (from `parotid_presence_trainval.csv`, TRAIN+VAL only — no test leakage)

both=208, only_R=110, only_L=112 (none=237 excluded).

- **Left-parotid pool = 320** (208 both + 112 only_L)
- **Right-parotid pool = 318** (208 both + 110 only_R)
- **Flip-symmetry pool = 638** side-instances (2×208 + 112 + 110)

Each per-side pool is ~50% more clean, gap-free data than the both-only 208 that clean-208/nnU-Net used.
All are subsets of the already-built `Dataset003_ParotidDirty` (430 patients) → datasets can be built by
**relabel+filter of existing NIfTI** (cheap I/O), no reconstruction from npz.

## 1. The four-way annotation-gap framing this completes

discard (clean-208, 0.8187) · penalise (dirty-430) · mask (masked-430, 0.7589) · **decompose (this)**.

## 2. Options considered

### Option A — two separate single-class nnU-Net experts  ★ RECOMMENDED (primary)
Left expert: 320 cases, foreground = left parotid only. Right expert: 318, right only. Standard single-class
nnU-Net `3d_fullres`, single fold, `nnUNetTrainer_250epochs_noMirror`. Combine at inference: left→label 2,
right→label 1.
- **Pros:** each is a *bog-standard* single-class segmentation → **nnU-Net runs it natively, no
  softmax→sigmoid / masking surgery** (the reason E6 couldn't use nnU-Net). Max clean data per side, **zero
  annotation-gap noise by construction**. Cleanest "decompose" story; reviewer-safe (no symmetry assumption).
  Upgrades the architecture from E6's custom U-Net (0.775 ceiling) to **nnU-Net (0.8187 ceiling)** — this is the
  real reason it could beat the champion where masked-430 could not.
- **Cons:** 2× training cost. No cross-side feature sharing. **Combine wrinkle:** an expert trained only on
  left glands may still fire on the (similar-looking) right gland → could mislabel it. *Mitigation:* resolve by
  hemifield (keep left-expert output where column > midline) and/or largest-CC per expected side; both are
  cheap post-hoc. Measure how often it happens (it doubles as a contralateral-behaviour probe).
- **Cost:** cheapest GPU, single fold, ~1.5–2.5 h/expert; run both in parallel on a 2-GPU pod. ~$3–6.

### Option B — flip-symmetry single expert  (cheapest; strong data pooling)
One expert on all **638** side-instances: every left gland + every horizontally-flipped right gland
(canonicalised to "left"). Inference: run for left; flip input → run → flip output for right.
- **Pros:** **half the cost** (one model), pools the **most** data (638), explicitly exploits parotid
  mirror-symmetry.
- **Cons / validity risk:** the whole project **disables mirror augmentation** (master §15) because it
  reintroduces the L/R bug. Building an expert *around* mirroring is thematically risky and a reviewer will
  scrutinise it — even though it's defensible here (the flip is **deterministic** and the side is **known** at
  inference, so there is no L/R confusion, unlike random mirror aug). Also assumes L/R task-symmetry; any
  systematic L/R appearance difference (asymmetric pathology, positioning, sub-diaphragmatic context) is lost.
- **Verdict:** attractive as a **cheap secondary** contrast (does pooling 638 beat two 318/320 experts?), but
  **not** the primary, for the reviewer-optics + symmetry-assumption reasons.

### Option C — single 2-class nnU-Net with ignore-label (native "mask" in one model)
Use nnU-Net's ignore-label so only-annotated glands contribute to the loss.
- **Fatal limitation:** you can only ignore the *entire contralateral hemifield* for a single-side case (the
  un-annotated gland's location is, by definition, unknown), which also discards true background signal. This
  is exactly the family **masked-430 already probed** (custom masked loss) → **0.7589, did not beat 0.8187.**
  Cleaner as native nnU-Net, but same conceptual ceiling. **Deprioritised** — E6 is the evidence.

## 3. Recommendation

**Primary: Option A (two separate single-class nnU-Net experts).** Cleanest decompose story, reviewer-safe,
native nnU-Net, and the architecture upgrade over E6 is the genuine reason it might beat 0.8187. If budget is
tight, **Option B alone** (one model, half cost) is the economical fallback and still pools the most data.

## 4. Eval plan (fairness stated explicitly)

Experts use **more total patient-instances** than clean-208 (320/318 vs 208) — that IS the point (it is a
**data-strategy** comparison, not a like-for-like architecture test). State it plainly. Evaluate on **both**:
- the **both-parotid 43** (combine experts → 2-label map, score via `pipeline/eval_testset.py` vs 0.8187 and
  the E6 arms), and
- the **single-side 58** (Part-1 machinery: annotated-side Dice + contralateral rate).
Report per-side Dice, HD95, Surface-Dice, and contralateral-prediction behaviour.

## 5. Expectation management (honest, up front)

E6 is a **yellow flag**: adding the single-side data via masking did **not** beat the both-parotid test. The
experts use the same extra data, so on the both-parotid 43 they may also land ~0.80–0.82 (competitive, maybe
not clearly > 0.8187). BUT E6's failure conflates *architecture weakness* (custom U-Net, 0.775) with *data
strategy*; Option A removes that confound by using nnU-Net, so a real chance remains. Their **likely** wins are
the **single-side eval** and **design cleanliness**, not necessarily a new SOTA on the easy 43. Do not oversell.

## 6. Execution checklist (when pod is up — awaiting budget approval)

1. Build `Dataset005_LeftExpert` (320, fg=left) + `Dataset006_RightExpert` (318, fg=right) by relabel+filter of
   `Dataset003_ParotidDirty` (hardlink images, rewrite labels to single foreground=1). Assert the 58 single-side
   + 43 both-parotid TEST cases are absent. **Also build/keep the Part-1 pod inputs** (`E7_singleside/nnraw`,
   `e6npz`) in the same upload.
2. `nnUNetv2_plan_and_preprocess` each; train `3d_fullres` single fold, `nnUNetTrainer_250epochs_noMirror`, `--npz`.
3. Predict both experts on the 43 both-parotid test + 58 single-side; combine (R=1, L=2, hemifield/CC tie-break).
4. Also run the four Part-1 predicts (nnU-Net clean-208, E4, dirty-430, masked-430) on the 58 while GPU is up.
5. Download all `preds/`; score locally (`eval_testset.py`, `E7_singleside/eval_singleside.py`, montage).
6. Log seed, exact commands, GPU, wall-time, **cost**. Read `POD_UPLOAD_PLAYBOOK.md` before uploading.

---

## 7. EXECUTION (as run, 2026-07-18)

Option A executed exactly as planned. `build_experts_pod.py` relabel+filtered `Dataset003_ParotidDirty` into
**`Dataset005_LeftExpert` (320 cases, fg=left)** and **`Dataset006_RightExpert` (318, fg=right)** — counts
match the verified TRAIN/VAL pools (§0), asserted no test leakage. Both are single-class nnU-Net `3d_fullres`,
**patch [48,224,192], batch 2, spacing 3.0×0.977×0.977 — identical config to the clean-208 baseline** (the
comparison differs only in data strategy, as intended). Trainer `nnUNetTrainer_250epochs_noMirror` (mirroring
OFF, master §15), single fold, `--npz`. Predict `--disable_tta`. Combine: `combine_experts.py` (right→label 1,
left→label 2, hemifield tie-break + largest-CC).

- **GPU / wall-time / cost:** 2× A40 (one expert per GPU, parallel). Both trained **21:22→~01:14 ≈ 3 h 52 m**,
  **~52.8 s/epoch** × 250. Predict+combine+full eval on-pod ~50 min. Pod total incl. the (troubled) upload
  ≈ **$8** of the $9.58 balance (training itself ≈ $3.4; the rest was the upload saga — see playbook note).
- **Expert internal validation Dice (own held-out fold-0):** left **0.8207**, right **0.8255**.
- Artifacts: `_pod_results/preds/{left,right}_{both,ss}`, `experts_{both,ss}`; checkpoints (reload-verified,
  epoch 250, 88 M params) + plans/dataset JSONs under `_pod_results/Dataset00{5,6}_*/`. Seeds = nnU-Net default.

## 8. RESULTS

### 8a. Both-parotid 43 (vs nnU-Net clean-208 = 0.8187) — the "easy" set
| model | Dice | Tversky | HD95 | SurfDice |
|---|---|---|---|---|
| nnU-Net clean-208 (baseline) | **0.8187** | 0.8166 | 5.25 | 0.8965 |
| **E7 experts (combined)** | 0.8099 | 0.8096 | 5.85 | 0.8896 |

Per-side (combined): R 0.8240 / L 0.7959. **Experts land ~0.81, competitive but do not beat 0.8187** — exactly
the honest expectation (§5). The extra per-side data does not help on the both-parotid subset.

### 8b. Single-side 58 — the realistic cases nobody had measured (annotated-side Dice+CC)
| model | Dice+CC | Tversky | HD95 | SurfDice | R | L | contra-rate |
|---|---|---|---|---|---|---|---|
| **E7 experts** | **0.8552** | 0.8456 | 4.07 | 0.9328 | 0.8449 | 0.8663 | 98.3% |
| nnU-Net clean-208 | 0.8523 | 0.8421 | 3.91 | 0.9325 | 0.8445 | 0.8607 | 96.6% |
| E4 custom clean-208 | 0.8011 | 0.7950 | 4.74 | 0.8847 | 0.7902 | 0.8128 | 100% |
| E6 masked-430 | 0.8010 | 0.8096 | 4.95 | 0.8820 | 0.7853 | 0.8179 | 94.8% |
| E6 dirty-430 | 0.7436 | 0.7306 | 13.01 | 0.8139 | 0.6961 | 0.7944 | 87.9% |

**The experts are the best on the single-side cases (0.8552), narrowly topping nnU-Net clean-208 (0.8523).**
Their real value is exactly here — the realistic cases + design cleanliness — not the easy both-parotid set.

### 8c. Individual experts (standalone, scored where their gland is annotated)
| expert | both-43 | single-side |
|---|---|---|
| Left | 0.7958 | **0.8663** (only_L, n=28) |
| Right | 0.8241 | **0.8449** (only_R, n=30) |

**Combining is lossless:** combined per-side (R 0.8240 / L 0.7959 on both-43) equals each standalone expert
(right 0.8241 / left 0.7958); on single-side the combined annotated-side Dice equals the standalone expert
exactly. The hemifield merge keeps each gland intact — **no cross-side interference** — which validates the
two-model decomposition.

## 9. VERDICT

- **Decompose ≈ discard, and both ≈ the champion.** On both-parotid, experts (0.8099) ≈ clean-208 (0.8187 nnU-Net
  / 0.775 E4). The four-way gap treatment is complete: discard 0.8187 · penalise (dirty) · mask 0.7589 ·
  **decompose 0.8099** — no strategy beats simply discarding the single-side data on the both-parotid test.
- **But the experts win on the single-side cases (0.8552, top),** the cases the whole study never scored — the
  genuine contribution. And the at-risk annotated gland is segmented *well* there (~0.85, higher per-gland than
  the both-side average), so single-side is not the catastrophe the clinical framing feared.
- **Reviewer-safe & honest:** experts use more patient-instances (320/318 vs 208) — a data-strategy result, not
  a like-for-like architecture win, and stated as such. Do not oversell as a new SOTA.
