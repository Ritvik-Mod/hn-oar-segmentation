# E7 — SINGLE-SIDE EVALUATION (free) + PER-SIDE SPECIALIST EXPERTS (optional GPU)

> **One Claude Code agent runs this.** Two parts:
> **Part 1 (do first, ~$0, no training):** evaluate the existing models on the held-out **single-side** test
> patients — the realistic cases the whole study never measured.
> **Part 2 (optional, ~$3–6 GPU):** train **per-side specialist models** (one left-parotid expert, one
> right-parotid expert) — the cleanest annotation-gap treatment, and the best remaining shot at beating 0.8187.
>
> Read `AGENT_INSTRUCTIONS.md` (ablation ground rules) and the "READ FIRST" list before anything. Owner:
> Ritvik Mod. Budget: ~$9.62 RunPod at time of writing — confirm with owner.

---

## 0. READ FIRST (get full context — in this order)
1. **`MASTER_PROJECT_REFERENCE.md`** — the project brain. Focus: §3 (dataset, the annotation gap), §14 (nnU-Net
   pipeline + E4), §16 (the annotation-gap clinical finding), §20 (the whole ablation + §20.E the E6 result),
   §21 (masked loss — tested, did not beat the best), §22 (verified numbers + artifact map).
2. **`PROJECT_NARRATIVE.md`** — canonical numbers/framing. **Never quote "0.62 → 0.82"**; honest figure is
   0.743 → 0.819 (+0.075) on the locked test set.
3. **`ablation_study/SYNTHESIS.md`** (esp. §4 annotation gap, §9 limitations — note §9.8: "the test set is the
   easy subset; nothing here measures single-side performance" — THIS is the hole Part 1 fills) and its
   **ADDENDUM — E6**.
4. **`ablation_study/E6_masked_loss/RESULT.md`** + **`ablation_study/E6_masked_loss_tasks.md`** — the E6 code
   and result you build directly on (preprocessing, dataloader, eval, per-case masks, the E4 custom U-Net).
5. **`ablation_study/E4_custom_3d_unet/RESULT.md`** and the nnU-Net pipeline scripts in `pipeline/`
   (`build_volumes.py`, `make_nnunet_dataset.py`, `fix_labels_qc.py`, `eval_testset.py`,
   `nnUNetTrainer_250epochs_noMirror.py`).
6. **`POD_UPLOAD_PLAYBOOK.md`** (repo root) — **read before any pod upload**: 8-stream chunked rsync with
   `--files-from` and `-rt` gives ~10× throughput; never `-a`/`-H` on macOS; share case IDs so duplicates can
   be `ln -f`'d; pods bill during upload; preprocess locally and upload only the compact cache.
7. Data sources you will reuse (already on the Mac, do not rebuild):
   - `patient classification/dataset_split.json` — the locked split (seed 42): **only TEST-split patients may
     be evaluated; only TRAIN/VAL patients may be trained on.** No leakage, ever.
   - `ablation_study/E2_annotation_gap/parotid_presence_trainval.csv` — per-patient both/only_R/only_L flags
     (TRAIN/VAL). For TEST-split presence, derive from the split + labelmaps.
   - `ablation_study/E2_annotation_gap/nnUNet_raw/Dataset003_ParotidDirty/` — 430 parotid-bearing TRAIN/VAL
     patients as 3D NIfTI (labels 1=R, 2=L). `ablation_study/E6_masked_loss/preprocessed/` — compact cache.

---

## PART 1 — SINGLE-SIDE EVALUATION  (do first; ~$0, CPU/local or a brief cheap pod)

**Why:** every number in the study so far is on the 43 **both-parotid** QC-clean test cases — the "easy"
subset. The locked TEST split also contains **~58 single-side patients** (expected ≈ only-R 30 + only-L 28 —
**verify from `dataset_split.json` + presence**) that were **never trained on**. These are the realistic,
clinically-common cases. On a one-sided tumour the annotated side IS the at-risk side, so scoring it is
exactly the clinically-relevant question.

**Steps:**
1. From `dataset_split.json` TEST list, identify the single-side patients (exactly one parotid annotated),
   excluding any QC-corrupt (reuse the `fix_labels_qc.py` 3×-median rule). Verify the count and composition;
   record it. **Confirm none overlap the training pools of any model being evaluated** (they shouldn't — they
   are TEST split — but assert it).
2. Build them into 3D volumes + single-side labelmaps (reuse `build_volumes.py` / the E6 preprocessing; same
   spacing/normalization as each model expects).
3. Run **every existing model** on them and score **the annotated side only** (per-side Dice / clinical
   Tversky / HD95 / Surface-Dice via `eval_testset.py` logic, single class):
   - nnU-Net clean-208 (`Parotid-Project/Results/...noMirror_FINAL` or the ablation copy),
   - E4 custom clean-208, E6 dirty-430, E6 masked-430,
   - and the Part-2 experts, if trained.
4. **Mechanism figure — contralateral-prediction rate:** on each single-side patient, measure whether each
   model *also* predicts the un-annotated (healthy) side, and report the **% of single-side cases where it
   does**, per model. Higher = less annotation-gap suppression, and (per §16) is **anatomically correct**, not
   an error. Expectation: masked-430 / experts predict the contralateral gland more often than dirty-430. Save
   a small montage (a couple of single-side cases, each model's prediction overlaid) illustrating this.
5. Output to `ablation_study/E7_singleside/`: `eval_singleside.csv` (per-model, per-side metrics),
   `contralateral_rate.csv`, `RESULT.md`, and the montage. **No GPU needed** for inference on the small custom
   U-Net; nnU-Net inference on CPU is slow but fine for ~58 cases (or a ~30-min cheap pod, <$0.50).

**Interpretation to record honestly:** on the annotated side, models may score *similarly* — the gap's effect
shows up more in the contralateral-prediction rate (the suppression figure) than in a big Dice gap. Say so.

---

## PART 2 — PER-SIDE SPECIALIST EXPERTS  (optional; decide AFTER seeing Part 1; ~$3–6 GPU)

> **Name it correctly: "per-side specialist models" / "side-decomposition ensemble" — NOT "mixture of
> experts"** (there is no gating network; a reviewer expecting a learned router will object). This is also
> unrelated to the dataset-routing MoE rejected earlier in the project (master §14 / CASE_FILE).

**The idea:** two independent **single-foreground-class** 3D segmentation models — a **left-parotid expert**
trained on every TRAIN/VAL patient with a left parotid, a **right-parotid expert** on every one with a right
parotid. At inference, run both and concatenate (left→label 2, right→label 1) into the standard labelmap.

**Why it's the best remaining training idea:** each expert is a *standard single-class* segmentation, so
**nnU-Net runs it natively — no softmax→sigmoid / masking surgery** (the reason E6 avoided nnU-Net). You get
nnU-Net-quality training AND the maximum clean data per side, with zero annotation-gap noise by construction.
It completes the four-way gap treatment: **discard** (clean-208) / **penalise** (dirty-430) / **mask**
(masked-430) / **decompose** (this).

**Data (verify before building):** TRAIN/VAL pool both=208, only_R=110, only_L=112 ⇒
**left-expert ≈ 320** (208 + 112), **right-expert ≈ 318** (208 + 110) — each ~50% more clean data than the
both-only 208. All TRAIN/VAL; no test leakage.

**⭐ FIRST, CRITIQUE AND IMPROVE THE DESIGN (do this before spending GPU):** think hard about whether the
plain two-model design is best, and write your reasoning into `RESULT.md` before executing. Consider at least:
- **Shared vs separate backbone:** masked-430 (a shared 2-channel model) already used the same per-side data;
  a shared backbone can exploit bilateral similarity + the hflip aug. Are two *fully separate* experts better
  (no cross-side interference) or worse (no feature sharing)? Would a single 2-class nnU-Net trained with
  nnU-Net's **region-based / ignore-label** mechanism achieve the "decompose" benefit in ONE model at half the
  cost? Evaluate that alternative explicitly.
- **Left/right symmetry trick:** since parotids are near-mirror-images, could a SINGLE expert trained on all
  ~638 side-instances (left + horizontally-flipped-right) serve both sides at inference (flip the input for
  the right)? That would halve training cost and pool the most data. Assess feasibility/validity.
- **Eval fairness:** experts use more total patient-instances than clean-208 — that's the point (it's a
  data-strategy comparison), but state it. Compare on BOTH the both-parotid 43 AND the single-side 58.
- **Expectation management (be honest in RESULT.md):** E6 is a yellow flag — adding single-side data (via
  masking) did *not* help the both-parotid test. Experts may likewise land ~0.80–0.82 on both-parotid
  (competitive, maybe not clearly beating 0.8187); their likely real win is the **single-side eval** + design
  cleanliness. Do not oversell.

Pick the design your critique supports (default: two nnU-Net single-class experts; but if the flip-symmetry
single-expert or region-based single-model is clearly better/cheaper, do that and justify it).

**Default execution (two nnU-Net experts):**
1. Build `Dataset005_LeftExpert` (≈320, foreground = left parotid only) and `Dataset006_RightExpert` (≈318,
   right only) as nnU-Net raw NIfTI — reuse `build_volumes.py` + a filtered `make_nnunet_dataset.py`
   (single-class labelmap). Keep the SAME held-out test patients (both-parotid 43 + single-side 58) out of
   training. Preprocess **locally**, upload only the compact preprocessed cache (see playbook).
2. `nnUNetv2_plan_and_preprocess` each; train `3d_fullres` **single fold, `nnUNetTrainer_250epochs_noMirror`,
   `--npz`** for each expert (mirroring OFF — master §15).
3. Predict both experts on the test images; **combine** into a 2-label map (R=1, L=2); evaluate with
   `pipeline/eval_testset.py` on the both-parotid 43, and per-side on the single-side 58 (Part 1 machinery).
4. Compare vs nnU-Net clean-208 (0.8187) and the E6 arms. Report per-side, HD95, Surface-Dice, and the
   contralateral-prediction behaviour.

**Budget discipline:** **try HPC (rachel L40) first for $0.** Else cheapest RunPod GPU (4090/A5000), single
fold, ~1.5–2.5 h/expert. Preprocess locally + upload compact cache per the **POD_UPLOAD_PLAYBOOK.md**;
terminate the pod the moment train+download finish; download BEFORE terminating. Run the two experts in
parallel on a 2-GPU pod if cheaper. Expected ~$3–6 total.

---

## OUTPUTS & DOC-UPDATING
- All new files under **`ablation_study/E7_singleside/`** (Part 1) and **`ablation_study/E7_experts/`**
  (Part 2): `RESULT.md`, eval CSVs, checkpoints (or pointer into `_pod_results_*`), logs, figures, and the
  design-critique writeup.
- **Master doc:** append under **`MASTER_PROJECT_REFERENCE.md` §20.F** (a stub is there). Append-only; do NOT
  edit §1–§19 or the frozen §20.A/B/C/E.
- **`ablation_study/SYNTHESIS.md`:** add an "ADDENDUM — E7" at the end (don't rewrite existing text); update
  §9.8's "nothing measures single-side" caveat once Part 1 lands.
- **Read-only zones (never modify):** existing `checkpoints/`, `ML_Dataset_Final`, split JSONs, `dataset.h5`,
  frozen historical docs, web-app/showcase, and the frozen E1–E6 results.
- Record seed, exact commands, GPU, wall-time, and **cost** in every `RESULT.md`.

## HANDOFF PROTOCOL (standing instruction — same as E6)
**When the owner types "rdy for handoff", STOP and output one self-contained handoff prompt** (copy-paste code
block) so a fresh agent can continue with zero context loss. Include: (1) what was accomplished this session
(Part 1 done? experts trained? exact metrics + file paths), (2) exact current state (mid-run checkpoints/epochs
+ how to resume; pod status + remaining balance), (3) what's left, in order, with commands, (4) gotchas hit
(data paths, leakage checks, upload, GPU/OOM), (5) instruction to paste Appendix A + this state into the new
agent. Also **proactively warn the owner when you sense you're low on context/tokens** so nothing is lost
mid-training.

---

## APPENDIX A — NEW-AGENT KICKOFF PROMPT (owner pastes this to start a fresh agent)

```
You are a Claude Code agent taking over an ML research task in the "ct scan project" folder (head-and-neck CT
parotid-gland segmentation). Verify claims against raw data (CSVs/checkpoints); do not trust prose over files.

STEP 1 — GET FULL CONTEXT (read before doing anything, in this order):
  1. MASTER_PROJECT_REFERENCE.md — the project brain. Focus §3, §14, §16 (annotation gap), §20 (ablation +
     §20.E E6 result), §21 (masked loss — tested, didn't beat the best), §22 (verified numbers + artifacts).
  2. PROJECT_NARRATIVE.md — canonical numbers/framing. NEVER quote "0.62 → 0.82"; honest = 0.743 → 0.819.
  3. ablation_study/SYNTHESIS.md (incl. ADDENDUM — E6) — findings + limitations (§9.8: single-side never
     measured — that is the hole you're filling).
  4. ablation_study/E7_singleside_eval_and_experts_tasks.md — YOUR TASK. Read it fully: Part 1 (single-side
     eval, free) and Part 2 (per-side specialist experts, optional GPU). It defines data sources, the
     leakage rules, the design critique you must do first, evaluation, outputs, budget, and the handoff protocol.
  5. ablation_study/AGENT_INSTRUCTIONS.md — golden rules (locked test set; eval only via
     pipeline/eval_testset.py; append-only doc updates; read-only zones; nnU-Net always noMirror + --disable_tta).
  6. ablation_study/E6_masked_loss/ (RESULT.md + code) and ablation_study/E4_custom_3d_unet/ — what you build on.
  7. POD_UPLOAD_PLAYBOOK.md — MUST read before any pod upload (8-stream chunked rsync; no -a/-H on macOS;
     preprocess locally + upload only the compact cache; pods bill during upload).

STEP 2 — HOW TO WORK IN THIS FOLDER:
  - Only TEST-split patients (patient classification/dataset_split.json) may be evaluated; only TRAIN/VAL
    patients may be trained on. Assert no leakage. Reuse Dataset003 / E6 preprocessed data; don't rebuild.
  - Score everything with pipeline/eval_testset.py. Put new outputs under ablation_study/E7_singleside/ and
    ablation_study/E7_experts/. Copy scripts before editing; never edit frozen originals or E1–E6 results.
  - Do Part 1 first (free). For Part 2, FIRST write a design critique (shared-vs-separate backbone;
    flip-symmetry single expert; region-based single model) then execute the best option. Name it "per-side
    specialist models", NOT "mixture of experts".
  - Budget ~ (confirm with owner) on RunPod; try HPC (rachel L40) first for $0; cheapest GPU, single fold,
    terminate promptly. Log cost.

STEP 3 — EXECUTE from wherever it stands (owner pastes prior-session state below). Update
ablation_study/E7_*/RESULT.md, MASTER_PROJECT_REFERENCE.md §20.F, and SYNTHESIS.md as you go. When the owner
types "rdy for handoff", follow the handoff protocol in the task doc.

[PASTE PRIOR-SESSION STATE HERE — or "fresh start, nothing done yet" if this is the first agent.]
```
