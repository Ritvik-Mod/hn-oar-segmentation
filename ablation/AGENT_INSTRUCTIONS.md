# AGENT INSTRUCTIONS — how to execute the ablation study and keep the docs live

> For the Claude Code agent(s) working on this repo. Read this FIRST, then your track file
> (`TRACK_A_nnUNet_tasks.md` **or** `TRACK_B_custom_tasks.md`), and keep
> `MASTER_PROJECT_REFERENCE.md` + `ABLATION_PLAN.md` open as context.

---

## 1. ORIENTATION (do this before any work)
1. Read **`MASTER_PROJECT_REFERENCE.md`** (repo root) end to end — it is the full project brain: dataset,
   existing code, all prior results, every caveat. Do not re-derive things it already documents.
2. Read **`ABLATION_PLAN.md`** — the questions, the experiment matrix, and the ground rules.
3. Read your **own track file only** and execute only the experiments it assigns to you. The other track is
   handled by a *separate* Claude Code account working in parallel — **do not do the other track's tasks.**

## 2. WHICH AGENT ARE YOU?
- If you were given **`TRACK_A_nnUNet_tasks.md`** → you are **Track A (nnU-Net)**. You own E1, E2, E5.
- If you were given **`TRACK_B_custom_tasks.md`** → you are **Track B (custom PyTorch)**. You own E3, E4,
  A1, P0.
- The two tracks share **no** data or code dependencies. You never need to wait for the other agent.

## 3. GOLDEN RULES (never violate)
- **Test set is sacred and fixed:** score every model on the SAME locked 43-case test set
  (`Dataset002_Parotid/labelsTs`) via `pipeline/eval_testset.py`. Never train on test patients; never change
  the test set between experiments.
- **One variable per experiment.** Hold everything else identical to the stated baseline.
- **nnU-Net = no-mirror trainer + `--disable_tta` always** (mirroring reintroduces the L/R bug, master §15).
- **Read-only zones — do NOT modify:** existing `checkpoints/`, `ML_Dataset_Final/`, the split JSONs,
  `dataset.h5`, `pipeline/` scripts (copy, don't edit in place, if you need a variant), and **anything under
  the web-app / showcase folders** (`webapp/`, `demo/`, `Parotid-Project/Website/`,
  `Parotid-Project/Visualizations/`) — those are out of scope, ignore them entirely.
- **All new outputs** go under `ablation_study/<Eid>/` (see layout in the plan). Never scatter files.
- **Reproducibility:** every run records seed, exact command, GPU type, wall-time, and file/commit versions.
- **If a result is surprising or a plan step is blocked:** do NOT silently change scope or invent a
  workaround that breaks comparability. Log the issue in your `RESULT.md` + the status table with ⚠️, state
  your hypothesis, and continue with the next independent task.

## 4. DOC-UPDATING PROTOCOL (this is required, not optional)
You must keep two documents live as you work. To avoid the two agents colliding, **each agent edits only its
own regions**:

### 4a. `ABLATION_PLAN.md` — the status table (Section 5)
- Update **only the rows for the experiments your track owns.** Track A edits rows E1/E2/E5; Track B edits
  rows E3/E4/A1/P0. Do not touch the other track's rows.
- Transition each row: `⬜ not started` → `🟡 in progress` (when you start it, with a timestamp in Updated) →
  `✅ done` (fill Test Dice / HD95 / Notes) or `⚠️` (if blocked, with a one-line reason).

### 4b. `MASTER_PROJECT_REFERENCE.md` — Section 20 "ABLATION STUDY — LIVING LOG"
- A stub Section 20 already exists at the bottom with two subsections: **20.A (Track A log)** and
  **20.B (Track B log)**. **Append only under your own track's subsection** — never edit the other's, and
  never edit Sections 1–19 (the historical brain is frozen; the study only *appends*).
- For each experiment, append a short dated entry: what you ran, the command, the headline metrics, and a
  1–3 sentence interpretation. Also keep the Section-20 results table row for your experiments updated.
- When your track is fully done, add a one-line "Track X complete" marker under your subsection.

### 4c. Per-experiment `RESULT.md`
- Write `ablation_study/<Eid>/RESULT.md` for every experiment with: question, exact command(s), config,
  seed, GPU, wall-time, metric table, and interpretation. This is the primary record; the master-doc entry
  is the summary.

**Update cadence:** flip the status to 🟡 the moment you start an experiment; write `RESULT.md` + flip to ✅
and append the master-doc entry the moment it finishes. Don't batch updates to the end — the owner watches
these docs to track progress across both accounts.

## 5. ANTI-COLLISION (two accounts, possibly one synced folder)
- Only ever write inside `ablation_study/<your Eids>/`, and only your own rows/subsections in the two shared
  docs. Because the two agents touch **disjoint files and disjoint doc regions**, there should be no
  conflicts.
- If the folder is git-tracked and you hit a conflict, prefer append/rebase over overwrite, and never revert
  the other track's entries.
- Do not create or move top-level folders other than `ablation_study/` subfolders.

## 6. ENVIRONMENT NOTES
- **nnU-Net (Track A):** register the custom trainer on every fresh pod (see master §14.4 /
  `HANDOFF_5fold_ensemble.md` STEP 2). Always `--npz` for training; always `--disable_tta` for predict. Env
  vars `nnUNet_raw / nnUNet_preprocessed / nnUNet_results`.
- **Custom PyTorch (Track B):** reuse the existing model/dataloader/loss code (master §9–§11); copy a script
  before modifying it (don't edit the frozen originals in place). HPC (`rachel`, 2×L40, PBS) is free when
  GPUs aren't contended; else a cheap cloud GPU (4090/A5000). CPU is fine for eval/analysis (P0, A1).
- **Compute discipline:** these are small experiments — single fold unless stated. Don't launch 5-fold or
  expensive sweeps unless the plan asks. Note approximate cost in `RESULT.md`.

## 7. DEFINITION OF DONE (per track)
- Every experiment your track owns has: a populated `ablation_study/<Eid>/RESULT.md`, a filled status row in
  `ABLATION_PLAN.md`, and an appended entry in master §20 under your subsection.
- Track A done = E1, E2, E5 logged. Track B done = E3, E4, A1, P0 logged.
- **Do NOT start S1 (synthesis)** unless you can see that *both* tracks are ✅ in the status table. Whichever
  agent finds both tracks complete may do S1 (write `ablation_study/SYNTHESIS.md` + the unified table in
  master §20); otherwise leave S1 for the owner.
