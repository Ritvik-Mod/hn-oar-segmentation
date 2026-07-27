# E7 — RunPod run sheet (Part 1 inference + Part 2 two-expert training, one session)

**For: Ritvik.** One pod session finishes **everything left in E7**: trains the two per-side experts, and runs
all Part-1 single-side inference (the 4 existing models + the experts). Budget approved: **~$9.58**. Try the
**free HPC (rachel L40) first**; else cheapest **2-GPU** RunPod (2× RTX 4090 or 2× A5000 — 4090 caps at 2/pod).

**What you actually do:** (1) rent pod, (2) `[MAC]` run the upload block, (3) `[POD]` run one script in tmux,
(4) `[MAC]` run the pull-back block + tell me. Everything else is automated in `pod_run_e7.sh`.

**Est. cost:** upload ~1 h (do it on a $0.25/hr network-volume pod, or eat ~$1 on the GPU box) + preprocess
~20 min + train ~2–2.5 h/expert **in parallel** (2 GPUs → ~2.5 h wall) + predict/eval ~20 min ≈ **~$4–7**.

---

## Everything is already staged on the Mac
`ablation_study/E7_experts/_pod_upload/` is one hardlinked tree (15 G apparent, ~0 real) in the exact
`/workspace` layout. Datasets **005/006 are built on the pod** from Dataset003 (`build_experts_pod.py`) — no
image is uploaded twice. Contents: Dataset003 (430, 11 G) · E7 single-side `nnraw`+`e6npz` (58) · Dataset002
both-parotid test (43) · 4 model checkpoints · all scripts + the custom trainer.

## STEP 1 — rent the pod
- **2 GPUs** (2× RTX 4090 / A5000 / L40S). 3d_fullres here ≈ patch [48,224,192] @ b2 (~30 M params) — measured
  ~30 s/epoch on a 4090, ~35 s on L40S ⇒ 250 epochs ≈ **~2–2.5 h/expert**, run in parallel.
- **Volume Disk: set 300 GB explicitly** (preprocessed ≈ 190 MB/case × 638 ≈ 120 G + raw + overlay).

## STEP 2 — `[MAC]` upload (8-stream chunked, per POD_UPLOAD_PLAYBOOK.md)
```bash
export POD=root@<HOST>; export PORT=<PORT>; export KEY=~/.ssh/id_ed25519
ssh -p $PORT -i $KEY $POD 'mkdir -p /workspace'
cd "ablation_study/E7_experts/_pod_upload"
find . -type f ! -name '.DS_Store' ! -name '._*' > /tmp/e7.txt; sort -o /tmp/e7.txt /tmp/e7.txt
total=$(wc -l < /tmp/e7.txt); per=$(( (total + 7) / 8 )); rm -f /tmp/e7c_*; split -l $per /tmp/e7.txt /tmp/e7c_
for c in /tmp/e7c_*; do ( for try in $(seq 1 8); do \
  rsync -rt --partial --timeout=120 --files-from="$c" \
    -e "ssh -p $PORT -i $KEY -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes" \
    . "$POD:/workspace/" && break; echo "[$c] retry $try"; sleep 5; done ) & done; wait
echo "UPLOAD DONE"
# verify counts match the Mac
ssh -p $PORT -i $KEY $POD 'echo D3=$(ls /workspace/nnunet/nnUNet_raw/Dataset003_ParotidDirty/imagesTr/*.nii.gz|wc -l) \
  SS=$(ls /workspace/E7/nnraw/imagesTs/*.nii.gz|wc -l) NPZ=$(ls /workspace/E7/e6npz/*.npz|wc -l) \
  D2=$(ls /workspace/E7/Dataset002_test/imagesTs/*.nii.gz|wc -l)'   # expect D3=430 SS=58 NPZ=58 D2=43
```

## STEP 3 — `[POD]` run everything in tmux
```bash
pip install -q nnunetv2 nibabel scipy   # restart wipes pip; re-run after any Stop
export S=/workspace/E7/scripts
tmux new -s e7
bash /workspace/E7/scripts/pod_run_e7.sh 2>&1 | tee /workspace/E7/run.log
# Ctrl-b d to detach. Watch training: tail -f /workspace/E7/train_left.log /workspace/E7/train_right.log
```
`pod_run_e7.sh` is resumable (guards on `checkpoint_final.pth`) and self-checks inputs. It prints the headline
experts numbers (both-43 + single-side-58) at the end.

## STEP 4 — `[MAC]` pull back, then I finish scoring locally
```bash
mkdir -p ablation_study/E7_experts/_pod_results
# predictions for every model (experts + the 4 Part-1 models)
rsync -rt --partial -e "ssh -p $PORT -i $KEY" "$POD:/workspace/E7/preds" ablation_study/E7_experts/_pod_results/
# the two trained expert checkpoints (fold_0: checkpoint_final.pth + plans.json + dataset.json)
for d in Dataset005_LeftExpert Dataset006_RightExpert; do
  rsync -rt --partial -e "ssh -p $PORT -i $KEY" \
    "$POD:/workspace/nnunet/nnUNet_results/$d" ablation_study/E7_experts/_pod_results/
done
rsync -rt --partial -e "ssh -p $PORT -i $KEY" "$POD:/workspace/E7/*.log" ablation_study/E7_experts/_pod_results/
# VERIFY before terminate (playbook §4): diff file inventory pod-vs-local, then Terminate.
```

## STEP 5 — `[MAC]` final scoring (light, local)
- `pipeline/eval_testset.py` on `experts_both` vs Dataset002 labelsTs → experts on the **both-parotid 43** (vs 0.8187).
- `ablation_study/E7_singleside/eval_singleside.py` (point its pred dirs at `_pod_results/preds/*`, add an
  `experts_ss` row) → annotated-side Dice + **contralateral rate** for all 5 models on the **58 single-side**.
- `ablation_study/E7_singleside/make_montage.py --cases <a b c>` → the suppression montage.
- I then write results into both RESULT.md files, master §20.F, and the SYNTHESIS ADDENDUM.

## Golden rules honoured
Only TRAIN/VAL patients train (experts = 320/318, asserted); only TEST patients scored (43 + 58); nnU-Net
always noMirror trainer + `--disable_tta`; new outputs only under `E7_experts/` and `E7_singleside/`. Log seed,
commands, GPU, wall-time, **cost** into RESULT.md when done.
