# Track A — RunPod run sheet (finishes E1 + E2 + E2b in one go)

**For: Ritvik. Written 2026-07-16 by the Track A agent.**

**This finishes ALL of Track A's GPU work in a single pod session.** Nothing is left over: E1, E2 and E2b
train, predict, and evaluate from one script. When it's done you send the results back and I close out every
RESULT.md, the status table, and master §20.A.

There are **4 things you actually do**. Everything else is automated.

| | Where | What |
|---|---|---|
| 1 | RunPod | rent a 4090 |
| 2 | `[MAC]` | run the upload block |
| 3 | `[POD]` | run **one** script in tmux, walk away (~5–8 h) |
| 4 | `[MAC]` | run the pull-back block, tell me it's done |

**Cost: ~5–8 GPU-hours ≈ $3–6 on a 4090.**

---

## Everything is already built — no dataset work needed

All three nnU-Net raw datasets are built and verified on your Mac:

| Dataset | What it is | Cases | Size |
|---|---|---|---|
| `Dataset002_Parotid` | the **clean** baseline (already existed at `Parotid-Project/Datasets/`) | 208 train / 43 test | 6.5 GB |
| `Dataset003_ParotidDirty` | **E2** — clean 208 + 222 single-side partial-label patients | 430 train / 43 test | 12 GB |
| `Dataset004_ParotidGapped` | **E2b** — same 208 patients, gap simulated at constant N | 208 train / 43 test | 30 MB* |

\* tiny locally because its images are hardlinks to Dataset002; the `tar -h` below expands them properly.

All three share the **same 43-case locked test set** (byte-identical clones), so every number is comparable.

---

## STEP 1 — rent the pod

- **GPU:** a single **RTX 4090** (or A5000). `3d_fullres` here is patch [48,224,192] @ batch 2, ~30M params —
  the original single-fold run took ~1 h on an H100, so budget ~1.5–3 h per run on a 4090.
- **Template:** PyTorch 2.x · **Disk:** **≥ 80 GB** (datasets ~19 GB + nnU-Net preprocessed + checkpoints).
- Prefer a **network volume** — it's what saved the final checkpoint when the balance ran out last time
  (master §17.1). With one, a pod stop won't lose the run.

## STEP 2 — `[MAC]` upload

> ## ⚠️ CORRECTED 2026-07-17 — the advice that was here was WRONG and cost hours.
> It recommended `rsync -aH`. **Do NOT use `-aH`.** On macOS (openrsync) `-H` defers every hardlinked file
> to a final pass that may never run — it burned an hour transferring nothing usable, leaving the target
> dataset empty. `-a` additionally poisons retry loops on a RunPod network volume (chown fails → non-zero
> exit → infinite retry). `--info=progress2` is GNU rsync 3.x only and is rejected outright.
>
> **The method that works, plus all measurements and traps, is in
> [`POD_UPLOAD_PLAYBOOK.md`](../POD_UPLOAD_PLAYBOOK.md) at the repo root. Read that first.**

**Measured:** single stream **~420 KB/s** → **8 chunked parallel streams ~4.3 MB/s (10×)**. `.nii.gz` is
already gzipped (re-gzip ratio 1.000) so `tar -z` is pointless, and `tar -h` inflates 12 GB → 25 GB by
dereferencing hardlinks. **Upload each unique file once; rebuild duplicates on the pod with `ln -f`.**

```bash
cd "/Users/ritvikmod/Desktop/Academics/Projects/ct scan project"
export POD=root@<POD_HOST>; export PORT=<POD_SSH_PORT>; export KEY=~/.ssh/id_ed25519

# 1. code first (instant)
scp -P $PORT -i $KEY pipeline/nnUNetTrainer_250epochs_noMirror.py \
             pipeline/eval_testset.py \
             ablation_study/pod_run_all_trackA.sh \
             $POD:/workspace/

# 2. Dataset002 only (~6.5 GB) — 8 chunked parallel streams. NOTE: -rt, NOT -a. No -H.
cd "Parotid-Project/Datasets"
ssh -p $PORT -i $KEY $POD "mkdir -p /workspace/nnUNet_raw"
find Dataset002_Parotid -type f \( -name '*.nii.gz' -o -name '*.json' \) | sort > /tmp/d2.txt
total=$(wc -l < /tmp/d2.txt); per=$(( (total + 7) / 8 )); rm -f /tmp/d2chunk_*
split -l $per /tmp/d2.txt /tmp/d2chunk_
for c in /tmp/d2chunk_*; do ( for try in $(seq 1 8); do \
  rsync -rt --partial --timeout=120 --files-from="$c" \
    -e "ssh -p $PORT -i $KEY -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes" \
    . "$POD:/workspace/nnUNet_raw/" && break; sleep 5; done ) & done; wait

# 3. Dataset003's NEW cases only (~5.8 GB) — same chunked loop over:
#      ls imagesTr/PAR1*.nii.gz labelsTr/PAR1*.nii.gz
#    dest: $POD:/workspace/nnUNet_raw/Dataset003_ParotidDirty/
# 4. Dataset004: labelsTr (29 MB) + jsons ONLY — its images are Dataset002's.
```

**Then rebuild every duplicate on the pod — instant, byte-exact.** (Dataset003/004 were deliberately built to
reuse Dataset002's *exact* case IDs so this works; verified 208/208 shared inodes locally.)
```bash
D2=/workspace/nnUNet_raw/Dataset002_Parotid
D3=/workspace/nnUNet_raw/Dataset003_ParotidDirty
D4=/workspace/nnUNet_raw/Dataset004_ParotidGapped
mkdir -p $D4/imagesTr $D4/imagesTs $D4/labelsTs $D3/imagesTs $D3/labelsTs $D3/labelsTr
for f in $D2/imagesTr/*.nii.gz; do ln -f "$f" $D4/imagesTr/$(basename $f); ln -f "$f" $D3/imagesTr/$(basename $f); done
for f in $D2/labelsTr/*.nii.gz; do ln -f "$f" $D3/labelsTr/$(basename $f); done
for f in $D2/imagesTs/*.nii.gz; do ln -f "$f" $D4/imagesTs/$(basename $f); ln -f "$f" $D3/imagesTs/$(basename $f); done
for f in $D2/labelsTs/*.nii.gz; do ln -f "$f" $D3/labelsTs/$(basename $f); ln -f "$f" $D4/labelsTs/$(basename $f); done
```

Cuts the upload from ~25 GB to ~12 GB. Re-run any chunk loop to resume — nothing is lost.

## STEP 3 — `[POD]` run the one script

```bash
ssh -p $PORT $POD          # or use the RunPod web terminal

ls /workspace/nnUNet_raw/  # must show: Dataset002_Parotid Dataset003_ParotidDirty Dataset004_ParotidGapped

tmux new -s trackA
bash /workspace/pod_run_all_trackA.sh 2>&1 | tee /workspace/trackA_run.log
```

With **3 GPUs** the script auto-detects them and runs all three experiments **in parallel**, one per GPU
(`CUDA_VISIBLE_DEVICES=0/1/2`) — wall-clock ≈ a single run rather than three. With fewer GPUs it falls back
to sequential automatically. Force sequential with `PARALLEL=0 bash /workspace/pod_run_all_trackA.sh`.

Detach with **Ctrl-b d**, reattach with `tmux attach -t trackA`. That's it — walk away.

**What the script does, in order:**

0. **Preflight** — checks the GPU is visible, installs any missing deps, and verifies all three datasets
   arrived with the right case counts (208/430/208 train, **43 test each**). Dies loudly on any mismatch, so
   a truncated upload can't silently corrupt a result.
1. **Registers the no-mirror trainer** into the nnU-Net package *and verifies the class actually resolves* —
   this is the step that must be redone on every fresh pod (master §14.4). If it silently failed you'd
   reintroduce the L/R mirror bug.
2. **Preprocesses** all three (`-d 2 -c 2d`, `-d 3 -c 3d_fullres`, `-d 4 -c 3d_fullres`), then **prints a
   plans comparison** — Dataset003/004 must land on the same spacing `[3.0, 0.977, 0.977]` and patch
   `[48, 224, 192]` as Dataset002, otherwise nnU-Net re-planned and that's a second variable (it flags this
   rather than hiding it).
3. **Trains** all three (fold 0, `--npz`, no-mirror). ~1.5–3 h each.
4. **Predicts** the locked test set with `--disable_tta`, asserting exactly 43 outputs each.
5. **Evaluates** all three against the **clean Dataset002 test labels**.
6. **Prints a summary** + the internal fold-0 val Dice as a mirror-bug canary.

**It is resumable.** If the pod drops or you re-run it: finished stages are skipped, and an interrupted
training resumes from `checkpoint_latest.pth` via `--c`. Re-running costs nothing but the unfinished work.

## STEP 4 — `[MAC]` pull results back

```bash
cd "/Users/ritvikmod/Desktop/Academics/Projects/ct scan project"
scp -P $PORT -r $POD:/workspace/results ablation_study/_pod_results
scp -P $PORT $POD:/workspace/trackA_run.log ablation_study/_pod_results/

# optional but recommended — keep the checkpoints before killing the pod
scp -P $PORT "$POD:/workspace/nnUNet_results/Dataset003_ParotidDirty/*/fold_0/checkpoint_final.pth" \
   ablation_study/E2_annotation_gap/
scp -P $PORT "$POD:/workspace/nnUNet_results/Dataset004_ParotidGapped/*/fold_0/checkpoint_final.pth" \
   ablation_study/E2_annotation_gap/checkpoint_final_gapped.pth
```

Then just tell me: **"pod results are in `ablation_study/_pod_results`"**.

I'll fill in `E1_2d_vs_3d/RESULT.md`, `E2_annotation_gap/RESULT.md` (both arms), the `ABLATION_PLAN.md`
status rows, and master §20.A — including the seed/GPU/wall-time record, which the script captures into
`results/run_manifest.txt`. Then Track A is **complete**.

---

## What each number will mean

| Run | vs | Reading |
|---|---|---|
| **E1** `2d` | clean 3d 1-fold = **0.8187** | ≥ ~0.75 → 3D buys little; the Phase-1 ceiling was preprocessing/labels. Collapse toward 0.62 → 3D is the driver. |
| **E2b** gapped-208 | **0.8187** | The clean isolation: same patients, same images, same N — **only labels differ**. *This is the number that measures the annotation gap.* |
| **E2** dirty-430 | **0.8187** | The realistic condition the Phase-1 baselines actually trained under. N also doubled, so a drop is a **lower bound** on the gap's cost. |
| **E2 vs E2b** | each other | Separates "label noise" from "more data" — what the 222 extra patients buy back. |

**Already settled without a GPU — E5 is done:** the 5-fold ensemble is **not** significantly better than a
single fold (+0.0015 Dice, paired Wilcoxon **p = 0.35**, worse on 17/43 cases). So ensembling explains ~none
of the 0.62 → 0.82 jump, leaving **{3D, preprocessing, labels}** — exactly what these three runs measure.
Preprocessing is then the remainder.

## Sanity checks (the script prints these — glance at them)

1. **Internal fold-0 val Dice ~0.79–0.80.** Mirror-ON was 0.4538, noMirror 0.7967 (master §15.1). Anything
   near 0.45 = the mirror bug is back → the trainer didn't register. Stop and tell me.
2. **43 predictions per run** — asserted automatically.
3. **Per-structure R/L.** Clean single fold was **R 0.8278 / L 0.8095**. A large R/L asymmetry in E2/E2b is
   itself a finding — the gap should bite the sides that were left un-contoured.

## If something breaks

| Symptom | Fix |
|---|---|
| `trainer did NOT register` | `nnUNetTrainer_250epochs_noMirror.py` didn't scp. Re-copy it to `/workspace/` and re-run the script. |
| `missing /workspace/nnUNet_raw/DatasetXXX` | the untar in Step 3 didn't run, or ran to the wrong path. |
| `has N test cases, expected 43` | truncated upload — re-scp the tarball. |
| CUDA OOM | rent a 4090/A5000 (24 GB). Don't shrink the patch/batch — that would break comparability with 0.8187. |
| pod died mid-training | just re-run the same script; it resumes from the latest checkpoint. |

Anything else: send me `trackA_run.log`.
