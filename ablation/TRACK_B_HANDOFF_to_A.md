# TRACK B HANDOFF — Agent B → Agent A (you are now Agent B)

Written 2026-07-17. Track B owns **P0, E4, A1, E3**. **P0/E4/A1 are ✅ DONE. Only E3 + the
final results-pull + S1 remain.** ⚠️ **NOTHING has been downloaded from the pod to the Mac yet**
— you MUST pull all models/predictions/figures at the end (commands in §5).

## 0. POD / ENV
- Pod: `root@64.247.206.241`, **port 14560** (⚠️ changes on every restart — get new one from RunPod
  Connect). Data on network volume at `/workspace/ctscan/`.
- Budget: ~$9 balance, ~$1.07/hr (verify). Keep buffer; don't leave the pod idle-billing.
- **pip deps are wiped on every pod restart** — after any restart:
  `pip install -q timm nibabel scipy scikit-image h5py scikit-learn matplotlib`
- macOS `._*` junk files rode in with earlier uploads and break nibabel globs — if any preprocess/glob
  step errors on `._PARxxxx`, run `find /workspace/ctscan -name '._*' -delete`.
- Mac vars to set each session:
  `export REPO="/Users/ritvikmod/Desktop/Academics/Projects/ct scan project"; export POD=root@64.247.206.241; export PORT=14560; export KEY=~/.ssh/id_ed25519`
- **Upload method that works** (Mac uplink ~4 MB/s, flaky): openrsync (macOS) REJECTS `--append-verify`
  and `--info=progress2`; use `-rt --partial --timeout=120` in a retry loop, or chunked `--files-from`
  with 8 parallel streams (see §4 Step 3). NEVER `-a` (chown fails on the network volume → poisons
  `&&` retry into an infinite loop). Single big file → `split -b 2000M` then chunked upload.

## 1. WHAT'S DONE (✅) — results already written into the repo docs
All three below have their **RESULT.md filled, ABLATION_PLAN.md row flipped ✅, and master §20.B +
§20.0 table updated** (local Mac repo). Numbers are final.

### P0 ✅ — 4 Phase-1 checkpoints on locked 43-case test (Phase-1 isotropic protocol, n=86 sides)
| Model | Dice | Tversky | HD95(iso) | SurfDice |
|---|---|---|---|---|
| Attention U-Net | 0.7434 | 0.7387 | 3.57 | 0.8827 |
| U-Net | 0.7390 | 0.7315 | 4.25 | 0.8702 |
| TransUNet | 0.7313 | 0.7307 | 5.60 | 0.8594 |
| Swin-UNet | 0.7156 | 0.7082 | 8.34 | 0.8794 |
- All cluster 0.72–0.74 (>0.62 val, <0.82 nnU-Net); architecture spread <0.03 Dice. Swin worst HD95
  but fewest one-sided misses (boundary problem, not missed glands → A1).
- ⚠️ TransUNet's checkpoint was truncated in the first upload (`1211352136` vs `1230961638`);
  re-uploaded + re-ran. **Pod `P0.../eval.csv` has only TransUNet+Swin rows** (from a `--only` rerun);
  the full 4-model per-case table is `P0.../per_case_full.csv` (173 lines). The **Mac repo already has
  the correct full `eval.csv` + `per_case.csv`** (I computed them).
- Pod files: `preds/` (172 npz = 4 models × 43, keys pred_r/pred_l/gt_r/gt_l) — **needed to recreate
  A1**; `per_case_full.csv`.

### E4 ✅ — hand-built plain 3D U-Net (the study's key experiment)
| Variant | Dice | Tversky | HD95 | SurfDice |
|---|---|---|---|---|
| Custom 3D U-Net (raw) | 0.7681 | 0.7746 | 26.76 | 0.8383 |
| **+ largest-CC postproc** | **0.7750** | 0.7788 | **6.05** | 0.8539 |
| ref nnU-Net 3d 1fold | 0.8187 | 0.8166 | 5.25 | 0.8965 |
- Scored with `pipeline/eval_testset.py` (true anisotropic) → directly comparable to nnU-Net (no HD95
  caveat). Trained fold-0 (166/42), early-stop epoch 83, ~50 min. **Thesis:** plain 3D on nnU-Net's
  pipeline reaches 0.77; largest-CC postproc collapses HD95 26.76→6.05 (≈nnU-Net) at ~constant Dice →
  **boundary gap = postproc not architecture**; residual −0.044 Dice = nnU-Net training machinery.
- Pod files: `ckpt_unet3d/best_model.pth` (the trained model), `pred_test/`, `pred_test_cc/`,
  `eval.csv` (2 rows), `training_history.json`, `preprocessed/` (~10 GB, regeneratable — DON'T pull).

### A1 ✅ — Swin boundary failure (Q5), Swin vs U-Net, 86 glands, true spacing
| Metric | Swin | U-Net | paired p |
|---|---|---|---|
| mean components | 2.84 | 1.20 | 1.3e-11 |
| % multi-component | 75.6% | 16.3% | — |
| mean stray islands | 1.85 | 0.23 | 1.1e-11 |
| farthest stray (mm) | 55.4 (max 306) | 12.2 | 1.9e-8 |
| HD95 full→largest-CC | 14.30→10.53 (−3.77) | 8.53→8.49 (−0.04) | — |
- **Both hypotheses confirmed:** H2 stray islands = dominant (removing them drops Swin HD95 3.77 mm vs
  0.04 for U-Net); H1 coarse 128→512 upsampling = residual (Swin-lcc 10.53 still >U-Net 8.49).
- Pod files: `A1.../summary.json`, `per_gland.csv`, `figures/montage_islands.png` (PAR0231 z=65),
  `figures/montage_coarse.png`. **(A1's analyzer was rewritten with bbox-crop — the current pod copy is
  the fast version; the Mac repo copy matches.)**

## 2. WHAT'S LEFT
1. **E3** — pretrained TransUNet + Swin. Setup in progress: the capped h5 (`trainval.h5`, 10.62 GB,
   34415 slices, 667/667 patients — verified) was **uploading via chunked streams to
   `/workspace/ctscan/h5parts/`** when this handoff was written. Finish per §4.
2. **Pull EVERYTHING back to the Mac** (§5) — models, preds, figures, CSVs, logs. Nothing pulled yet.
3. **Finalize E3 docs** after it runs: fill `E3.../RESULT.md` results table + interpretation, flip E3
   row ✅ in `ABLATION_PLAN.md`, update master §20.0 E3 rows + §20.B E3 entry, add "**Track B
   complete**" marker under §20.B.
4. **S1 (synthesis)** — only after E3 done AND you verify Track A's E1/E2/E5 are ✅ + "Track A complete"
   marker present in §20.A (per instructions). Write `ablation_study/SYNTHESIS.md` + master §20.C. S1
   MUST carry: (a) **val-vs-test caveat** (0.62 is Phase-1 *val*; P0 shows Phase-1 *test* ~0.74, so much
   of the headline 0.62→0.82 was measurement); (b) **preprocessing's share is a remainder** that E4
   measures directly; (c) E4's finding (plain 3D→0.775, +CC HD95 matches nnU-Net; residual=training
   machinery ⇒ architecture nearly irrelevant once pipeline is right); (d) A1's Swin explanation.

### Full results for the S1 unified table (all on the locked 43-case test set)
- nnU-Net 3d_fullres 1 fold: **0.8187** / HD95 5.25 / SurfD 0.8965 (reference)
- nnU-Net 5-fold ensemble (E5): 0.8202 — delta n.s. (p=0.35)
- nnU-Net 2d (E1): 0.8117 / 5.52 / 0.8943 (3D worth only +0.007)
- E2b gapped-208 (pure gap cost): 0.6899 / 9.63 / 0.7649 (−0.129 Dice)
- E2 dirty-430 (net gap cost): 0.7726 / 5.45 / 0.8400 (−0.046)
- P0 from-scratch 2D on test: Attn 0.7434 / U-Net 0.7390 / TransUNet 0.7313 / Swin 0.7156 (HD95 iso)
- E4 custom 3D: 0.7681 (raw) / 0.7750 (+CC, HD95 6.05)
- E3: **TBD** (TransUNet_pretrained vs P0 0.7313; Swin_pretrained vs P0 0.7156)
- A1: Swin boundary failure = stray islands (H2, dominant) + coarse upsample (H1, residual)

## 3. E3 SCIENTIFIC NOTES (for its RESULT.md)
- E3 evaluates on the 43-case test with `evaluate_e3.py` (reuses P0's `phase1_metrics`, **isotropic**),
  so it's apples-to-apples with the **P0 from-scratch** numbers (not with nnU-Net's aniso HD95).
- Compare: **TransUNet_pretrained vs P0 TransUNet 0.7313**, **Swin_pretrained vs P0 Swin 0.7156**. A rise
  = pretraining helped (answers Q3).
- ⚠️ **Two documented budget deviations** (state them): (1) the h5 caps empty slices at 40/patient
  (`--cap-empty-per-patient 40`) — slightly changes negative sampling; (2) training caps at 12000
  samples/epoch (`--samples-per-epoch 12000`) vs Phase-1's full pass. Absolute numbers carry these
  caveats; the pretrained-vs-scratch **delta** is the takeaway.
- ⚠️ **E3-Swin architecture caveat** (already in `E3.../RESULT.md`): a true same-arch pretrained Swin is
  impossible (frozen Swin window=8/512 vs ImageNet Swin-T window=7/224), so E3-Swin uses timm's Swin-T
  encoder + the frozen decoder — mild architecture confound on the Swin arm only. TransUNet arm is clean
  (258/258 ResNet-50 load). Lean on TransUNet for the headline pretraining claim.

## 4. E3 COMMANDS (finish the run)
**Step 4 — reassemble h5 (once `du -sh /workspace/ctscan/h5parts` ≈ 10.6 GB and the Mac upload's `wait`
returned) `[POD]`:**
```bash
cd /workspace/ctscan
rm -f trainval.h5
for p in $(ls h5parts/trainval.h5.part_* | sort); do cat "$p" >> trainval.h5 && rm "$p"; done && rmdir h5parts
stat -c%s trainval.h5      # must equal the Mac's `ls -l` of data/trainval.h5 (~10.62 GB)
python3 -c "import h5py; f=h5py.File('/workspace/ctscan/trainval.h5'); n=[0]; f.visititems(lambda k,o: n.__setitem__(0,n[0]+('image' in o))); print('image datasets:', n[0])"   # expect 34415
```
If size/count mismatch → re-run the Mac chunked upload (§ below), it resumes.

**Step 5 — train (budget config) `[POD]`:**
```bash
pip install -q timm
nvidia-smi --query-gpu=index --format=csv,noheader   # 1 or 2 GPUs?
cd /workspace/ctscan/ablation_study/E3_transformer_pretrain
tmux new -s e3
```
If **2 GPUs** (parallel):
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_e3.py --model transunet_pretrained --data /workspace/ctscan/trainval.h5 --split /workspace/ctscan/dataset_split.json --save-dir ./ckpt_transunet_pretrained --samples-per-epoch 12000 --epochs 40 --patience 8 > log_tr.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 train_e3.py --model swin_pretrained --data /workspace/ctscan/trainval.h5 --split /workspace/ctscan/dataset_split.json --save-dir ./ckpt_swin_pretrained --samples-per-epoch 12000 --epochs 40 --patience 8 > log_sw.txt 2>&1 &
wait
```
If **1 GPU** (sequential, one then the other):
```bash
python3 train_e3.py --model transunet_pretrained --data /workspace/ctscan/trainval.h5 --split /workspace/ctscan/dataset_split.json --save-dir ./ckpt_transunet_pretrained --samples-per-epoch 12000 --epochs 40 --patience 8
python3 train_e3.py --model swin_pretrained --data /workspace/ctscan/trainval.h5 --split /workspace/ctscan/dataset_split.json --save-dir ./ckpt_swin_pretrained --samples-per-epoch 12000 --epochs 40 --patience 8
```
Expect: `loaded 258/258` (TransUNet) / timm swin line, `samples/epoch: 12000 (capped)`, `Epoch 001/40`
with val loss dropping, early stop when converged (~2–4 h total). Watch: `tail -f log_tr.txt`.

**Step 6 — eval both `[POD]`:**
```bash
cd /workspace/ctscan/ablation_study/E3_transformer_pretrain
CUDA_VISIBLE_DEVICES=0 python3 evaluate_e3.py --model transunet_pretrained --ckpt ./ckpt_transunet_pretrained/best_model.pth --tag TransUNet_pretrained --device cuda
CUDA_VISIBLE_DEVICES=0 python3 evaluate_e3.py --model swin_pretrained --ckpt ./ckpt_swin_pretrained/best_model.pth --tag Swin_pretrained --device cuda
cat eval.csv
```

**(If the h5 upload was NOT finished at handoff — resume it) `[MAC]`:** the h5 parts are at
`$REPO/ablation_study/E3_transformer_pretrain/data/trainval.h5.part_*`. If they still exist:
```bash
cd "$REPO/ablation_study/E3_transformer_pretrain/data"
ls trainval.h5.part_* | sort > /tmp/h5parts.txt
n=$(wc -l < /tmp/h5parts.txt); per=$(( (n+7)/8 )); split -l $per /tmp/h5parts.txt /tmp/h5chunk_
ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new $POD "mkdir -p /workspace/ctscan/h5parts"
for c in /tmp/h5chunk_*; do ( for t in $(seq 1 12); do rsync -rt --partial --timeout=120 --files-from="$c" -e "ssh -p $PORT -i $KEY -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o StrictHostKeyChecking=accept-new" . "$POD:/workspace/ctscan/h5parts/" && break; sleep 5; done ) & done; wait
```
If the parts were already deleted/reassembled, rebuild from source (~16 min):
`python3 ablation_study/E3_transformer_pretrain/build_compact_h5.py --out ablation_study/E3_transformer_pretrain/data/trainval.h5 --splits train,val --cap-empty-per-patient 40`

## 5. PULL EVERYTHING BACK TO THE MAC (do before stopping the pod!)
Results + figures + logs (small):
```bash
cd "$REPO"
rsync -rt -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" --include='*/' --include='eval.csv' --include='per_case.csv' --include='per_case_full.csv' --include='per_gland.csv' --include='summary.json' --include='training_history.json' --include='*.txt' --include='*.png' --exclude='*' $POD:/workspace/ctscan/ablation_study/ ablation_study/_pod_results_B/
```
Models (checkpoints) — the trained networks (portfolio artifacts):
```bash
mkdir -p ablation_study/_pod_results_B/checkpoints
rsync -rt -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" \
  $POD:/workspace/ctscan/ablation_study/E4_custom_3d_unet/ckpt_unet3d/best_model.pth \
  $POD:/workspace/ctscan/ablation_study/E3_transformer_pretrain/ckpt_transunet_pretrained/best_model.pth \
  $POD:/workspace/ctscan/ablation_study/E3_transformer_pretrain/ckpt_swin_pretrained/best_model.pth \
  ablation_study/_pod_results_B/checkpoints/   # (rename each on arrival, they share a basename)
```
Predictions needed to RECREATE figures/analysis:
```bash
# P0 preds (172 npz) -> recreate A1 montages/analysis:
rsync -rt -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" $POD:/workspace/ctscan/ablation_study/P0_phase1_on_test/preds/ ablation_study/_pod_results_B/P0_preds/
# E4 test predictions (NIfTI, raw + CC) -> recreate E4 renderings:
rsync -rt -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" $POD:/workspace/ctscan/ablation_study/E4_custom_3d_unet/pred_test/ ablation_study/_pod_results_B/E4_pred_test/
rsync -rt -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=accept-new" $POD:/workspace/ctscan/ablation_study/E4_custom_3d_unet/pred_test_cc/ ablation_study/_pod_results_B/E4_pred_test_cc/
```
DON'T pull (regeneratable, huge): `trainval.h5`, `E4.../preprocessed/`, `h5parts/`, Dataset002, checkpoints/.
After pulling + verifying, **stop (don't terminate) the pod** to preserve balance + volume.

## 6. LOCAL REPO STATE (already committed by Agent B)
Updated on the Mac: `P0/RESULT.md`+`eval.csv`+`per_case.csv`; `E4/RESULT.md`+`eval.csv`+
`postproc_largestcc.py`; `A1/RESULT.md`; `ABLATION_PLAN.md` (P0/E4/A1 rows ✅); `MASTER_PROJECT_REFERENCE.md`
(§20.0 table + §20.B entries for P0/E4/A1). `train_e3.py` has the new `--samples-per-epoch` flag (also
uploaded to the pod). E3's RESULT.md still shows the pre-run staging text + the two-arm table with
`_pending GPU_` — fill it after E3 runs. Edit ONLY Track B regions (E3/E4/A1/P0 rows, master §20.B);
Track A's regions (§20.A, E1/E2/E5) are complete — don't touch.
