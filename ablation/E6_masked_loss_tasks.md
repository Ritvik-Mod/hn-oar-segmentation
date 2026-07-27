# E6 — MASKED PARTIAL-LABEL LOSS (single agent)

> **One Claude Code agent runs this.** Goal: close the loop on the project's star finding (the annotation
> gap) by showing that a **masked partial-label loss recovers the gap's cost** — training on all
> parotid-bearing patients while ignoring un-annotated glands in the loss. Read `AGENT_INSTRUCTIONS.md`
> (ablation ground rules) and the "READ FIRST" list below before doing anything.
>
> Owner: Ritvik Mod. This is a follow-on to the completed ablation (E1–E5, P0, A1, S1). Budget-constrained
> (~$3.90 RunPod) — see §5.

---

## 0. READ FIRST (get full context — do this before touching anything)
In this order:
1. **`MASTER_PROJECT_REFERENCE.md`** §10.2 (the masked loss), §14 (nnU-Net pipeline + E4 preprocessing),
   §20 (the whole ablation + the headline correction), §21 (why this experiment, the number-backed
   prediction), §22 (verified numbers + artifact map). This is the project brain.
2. **`PROJECT_NARRATIVE.md`** — the canonical numbers and framing (use these, never "0.62 → 0.82").
3. **`ablation_study/SYNTHESIS.md`** — §3 (decomposition), §4 (annotation-gap structure: clean−E2b=−0.129,
   E2−E2b=+0.083, clean−E2=−0.046), §8 (why masked loss is the best next step).
4. **`ablation_study/ABLATION_PLAN.md`** §1 (ground rules) and **`AGENT_INSTRUCTIONS.md`** (golden rules,
   doc-updating protocol).
5. **`ablation_study/E4_custom_3d_unet/RESULT.md`** + the E4 code/artifacts in
   **`ablation_study/_pod_results_B/E4_custom_3d_unet/`** (checkpoint `E4_custom_3d_unet_best.pth`,
   `norm_constants.json`, `training_history.json`) and `ablation_study/E4_custom_3d_unet/eval.csv`
   (0.7681 raw / 0.7750 CC). **You will reuse this exact model + preprocessing.**
6. **`pipeline/masked_loss.py`** (the loss, already written + unit-tested) and **`pipeline/eval_testset.py`**
   (the scorer — MUST be used for all final numbers).
7. **`ablation_study/E2_annotation_gap/nnUNet_raw/Dataset003_ParotidDirty/`** — the **430 parotid-bearing
   patients already reconstructed as 3D NIfTI** (imagesTr + labelsTr, labels 1=parotid_r, 2=parotid_l).
   **This is your training data source — do not rebuild it.** And
   `ablation_study/E2_annotation_gap/parotid_presence_trainval.csv` (per-patient only_R/only_L/both flags).

## 1. THE EXPERIMENT (three arms on the SAME custom 3D U-Net)
Hold the architecture, preprocessing, patch size, optimizer, schedule, seed, and **the locked 43-case test
set** identical to E4. Only the **training data scope** and the **loss** change:

| Arm | Train data | Loss | Status |
|-----|-----------|------|--------|
| **clean-208** | Dataset002 both-annotated (208) | ordinary Dice+BCE | ✅ EXISTS (E4 = 0.7681 / 0.7750 CC) — do not retrain |
| **dirty-430** | all parotid-bearing (430), single-side side left as background | ordinary Dice+BCE | ⬜ TRAIN |
| **masked-430** | all parotid-bearing (430) | **`masked_loss.py`** (mask the un-annotated channel) | ⬜ TRAIN — the key result |

**The claim this proves:** if `masked-430 > dirty-430` and `masked-430 ≥ clean-208`, the masked loss captures
the extra 222 patients' data benefit *without* the annotation-gap penalty. That is the fix for the star
finding. (Mirrors the E2/E2b logic — clean/dirty/gapped — but in the custom-U-Net world where per-channel
masking is possible; nnU-Net's softmax/labelmap formulation can't do it, which is why we use the custom
U-Net, see master §21.)

## 2. MASKED-LOSS INTEGRATION (the one subtle part)
- The custom 3D U-Net outputs **2 independent sigmoid channels** (ch0 = PAROTID_R, ch1 = PAROTID_L) — exactly
  the `[B, C, *spatial]` format `MaskedPartialLoss` expects. It should drop in with minimal glue.
- **Per-sample annotation mask `[B, 2]`:** for each patient/patch, `mask[:,0]=1` iff PAROTID_R is annotated
  for that patient, `mask[:,1]=1` iff PAROTID_L is. Derive from the labelmap (does label 1 / label 2 exist
  for that patient?) or from `parotid_presence_trainval.csv`. For a right-only patient the L channel is
  masked out of the loss entirely (target stays all-background but contributes nothing).
- **dirty-430 arm** uses the ordinary loss (un-annotated channel = background and IS penalised — this
  reproduces the gap). **masked-430 arm** uses `MaskedPartialLoss` (un-annotated channel ignored).
- Sanity-check the mask plumbing before a full run: confirm that flipping an un-annotated channel's target in
  the masked arm leaves the loss unchanged (the unit test in `masked_loss.py __main__` already shows this for
  the loss itself — verify it holds end-to-end through your dataloader).

## 3. EVALUATE + COMPARE (identical to the rest of the study)
- Predict the locked test set (Dataset002 `imagesTs`, 43 cases) and score with **`pipeline/eval_testset.py`**
  (3D Dice / Tversky / HD95 / Surface-Dice, true anisotropic spacing). Apply the same
  largest-connected-component post-processing E4 used (report raw AND +CC, like E4).
- Put all three arms in one table against the references: **E4 clean-208 (0.7681/0.7750 CC)**, and for
  context **nnU-Net 3d single fold 0.8187**. Note the n=43 noise floor (~0.02–0.05 cross-run) when
  interpreting — a masked-430 landing ~0.80–0.81 is a *mechanism confirmation*, not necessarily "beats
  nnU-Net".
- Also report on **single-side test behaviour if any** and per-side R/L Dice (the fix should most help the
  side that was under-trained).

## 4. OUTPUTS & DOC-UPDATING
- All new files under **`ablation_study/E6_masked_loss/`**: `RESULT.md`, `eval.csv`, checkpoints (or a
  pointer + keep them in `_pod_results_*`), training logs, and the mask-plumbing sanity check.
- **`RESULT.md`** must contain: the three-arm table, exact commands, seed/GPU/wall-time/**cost**, the
  mechanism verdict, and honest caveats (n=43, single fold/seed).
- **Update the master doc:** append your entries under **`MASTER_PROJECT_REFERENCE.md` §20.E** (a stub is
  there). Do NOT edit §1–§19 or the frozen §20.A/B/C — append only.
- **Update `ablation_study/SYNTHESIS.md`:** add a short "E6 — masked loss" addendum at the end (don't rewrite
  the existing text) and, if masked-430 changes the "best next step" story, note it.
- Do **not** touch: existing checkpoints, `ML_Dataset_Final`, split JSONs, `dataset.h5`, the frozen
  historical docs, or the web-app/showcase folders.

## 5. BUDGET DISCIPLINE (~$3.90 RunPod — tight)
- **Try the university HPC (rachel L40) FIRST** — the U-Net is small; if a GPU is free this costs $0 and
  saves the RunPod balance. Only fall back to RunPod if HPC GPUs are contended.
- On RunPod: cheapest adequate GPU (**RTX 4090 community ~$0.34/hr or A5000 ~$0.26/hr**), **single fold**,
  never 5-fold. Expected ~2–3 GPU-h per run.
- **Minimize billed upload:** preprocess the 430 volumes **locally** (CPU) into the compact cached-tensor
  format E4 used and upload only that — or attach the pre-existing network volume if it survived. Do NOT
  upload 13 GB of raw NIfTI to a running pod if avoidable (pods bill during upload; see
  `POD_UPLOAD_PLAYBOOK.md`).
- **Order to protect the budget:** run **masked-430 first** (the key result); terminate the pod; run
  **dirty-430 only if ≥ ~$1.5 remains**. clean-208 already exists (free). If budget runs out after
  masked-430, that alone + the existing E4 (0.768) + E2's nnU-Net gap evidence is a valid first result.
- **Terminate the pod the moment training + download finish.** Always download results BEFORE terminating.

---

## 6. HANDOFF PROTOCOL (standing instruction for this and any successor agent)
**When the user types "rdy for handoff" (or "ready for handoff"), STOP and output a single self-contained
handoff prompt** (in a copy-paste code block) so the user can start a fresh Claude Code agent with zero
context loss. It must contain, concisely:
1. **What was accomplished this session** — which arms trained, exact metrics obtained, files/checkpoints
   written (with paths), docs updated.
2. **Exact current state** — what is done vs partially done; if a training was mid-run, the checkpoint path,
   epoch reached, and how to resume; pod status (running/stopped/terminated) and remaining balance if known.
3. **What is left** — the specific remaining steps to finish E6, in order, with the exact commands.
4. **Gotchas hit** — anything that wasted time or needs care (data paths, mask plumbing, GPU/OOM, upload).
5. **The kickoff instructions** — tell the user to paste the "NEW-AGENT KICKOFF PROMPT" (Appendix A) plus the
   session-specific state above into the new agent.
Keep it factual and complete; assume the new agent has NOT seen this session.

Also: proactively **warn the user when you sense you are running low on context/tokens** ("we should hand off
soon") so nothing is lost mid-training.

---

## APPENDIX A — NEW-AGENT KICKOFF PROMPT (the user pastes this to start a fresh agent)

```
You are a Claude Code agent taking over an ML research task in the "ct scan project" folder (head-and-neck
CT parotid-gland segmentation). Work carefully and verify with the raw data; do not trust prose over CSVs.

STEP 1 — GET FULL CONTEXT (read before doing anything, in this order):
  1. MASTER_PROJECT_REFERENCE.md  — the complete project brain. Focus on §10.2 (masked loss), §14
     (nnU-Net pipeline + E4), §20 (the ablation study + the headline correction), §21 (this experiment's
     motivation), §22 (verified numbers + where every artifact lives).
  2. PROJECT_NARRATIVE.md — canonical numbers/framing. NEVER quote the retired "0.62 → 0.82"; the honest
     figure is 0.743 → 0.819 (+0.075) on the locked test set.
  3. ablation_study/SYNTHESIS.md — the study's findings (esp. §3, §4, §8).
  4. ablation_study/E6_masked_loss_tasks.md — YOUR TASK. Read it fully; it defines the three-arm experiment,
     the masked-loss integration, evaluation, outputs, budget discipline, and the handoff protocol.
  5. ablation_study/AGENT_INSTRUCTIONS.md — golden rules (locked test set; eval only via
     pipeline/eval_testset.py; append-only doc updates; read-only zones; never re-enable nnU-Net mirroring).
  6. ablation_study/E4_custom_3d_unet/RESULT.md + ablation_study/_pod_results_B/E4_custom_3d_unet/ — the
     custom 3D U-Net you will reuse (its code, preprocessing, norm_constants.json, checkpoint).
  7. pipeline/masked_loss.py and pipeline/eval_testset.py.

STEP 2 — HOW TO WORK IN THIS FOLDER:
  - The bulk data lives in ablation_study/E2_annotation_gap/nnUNet_raw/Dataset003_ParotidDirty/ (430
    parotid-bearing patients as 3D NIfTI, labels 1=R/2=L) — reuse it, do not rebuild.
  - Reuse E4's exact model + preprocessing so the comparison is clean; copy scripts before editing (never
    edit frozen originals in place). Put ALL new outputs under ablation_study/E6_masked_loss/.
  - Score every model on the SAME locked 43-case test set (Dataset002 labelsTs) with pipeline/eval_testset.py.
  - Budget is ~$3.90 on RunPod (confirm current balance with the user). Try the university HPC (rachel L40)
    first for $0; on RunPod use the cheapest GPU, single fold, run masked-430 first, terminate promptly.
    See POD_UPLOAD_PLAYBOOK.md to avoid billed-upload waste.

STEP 3 — EXECUTE E6 from wherever it currently stands (the user will paste the session state / what's already
done below this prompt). Update ablation_study/E6_masked_loss/RESULT.md, MASTER_PROJECT_REFERENCE.md §20.E,
and SYNTHESIS.md as you go. When I type "rdy for handoff", follow the handoff protocol in §6 of the task doc.

[PASTE PRIOR-SESSION STATE HERE — or "fresh start, nothing done yet" if this is the first agent.]
```
