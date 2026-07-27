# POD UPLOAD & GPU PLAYBOOK — hard-won lessons (Mac → RunPod)

**Written 2026-07-17 after the ablation study moved ~35 GB to rented pods across two accounts.**
Everything here is **measured on this Mac + RunPod**, not theory. Read before any future pod work — several
of these cost hours to learn. Reusable for any project, not just this one.

---

## 0. THE ONE-LINER

**Upload each unique file ONCE, via 8 parallel streams driven by chunked `--files-from`, using `-rt` (never
`-a`, never `-H`), and rebuild every duplicate on the pod with `ln -f`.**

---

## 1. THE MEASURED NUMBERS (Mac uplink → RunPod)

| Method | Throughput | Verdict |
|---|---|---|
| single `rsync`/`scp` stream | **~420 KB/s** | 12 GB would take **~8 h** |
| `xargs -P 8` (one rsync **per file**) | ~4.3 MB/s on **big** files; **~100 KB/s** on small ones | good for 27 MB files, terrible for 140 KB labels |
| **8 × chunked `--files-from`** | **~4.3 MB/s sustained, any file size** | ✅ **the answer — 10× a single stream** |

Bursts to 2.37 MB/s on a single stream prove **the line is not the cap** — it is per-stream TCP over a
long-haul link. **Parallel streams are the fix.** Don't go changing Wi-Fi.

---

## 2. THE FIVE TRAPS (each one bit us)

### 2.1 `rsync -H` is catastrophic on macOS — never use it
macOS ships **openrsync** ("protocol version 29, rsync 2.6.9 compatible"). Its `-H` **defers every
hardlinked file to a final linking pass**. We ran it for a full hour: it transferred **53 unique files and
NOTHING of the dataset we actually needed** (the destination dir was literally empty). Upload unique files
once and `ln -f` the duplicates on the pod instead.

### 2.2 `-a` poisons retry loops on a network volume
`rsync -a` implies `-o`/`-g` (preserve owner/group). RunPod's network volume **rejects chown** →
`chown ... failed: Operation not permitted` → **rsync exits non-zero even though every byte transferred** →
any `... && break` retry loop **retries forever**. **Use `-rt`** (recursive + mtime). That is all data files
need; mtime is what makes resume/skip work.

### 2.3 One SSH handshake PER FILE throttles small files
`ls files | xargs -P 8 -I{} rsync ... {} pod:dest/{}` opens a **new SSH connection per file**. Fine for
27 MB images (handshake amortised); **disastrous for 500 small files** (~100 KB/s). Chunked `--files-from`
gives **one handshake per chunk** — 8 total instead of 500.

### 2.4 openrsync rescans the whole file list on every retry
No incremental recursion. A retry loop that restarts rsync re-stats all ~500 files each cycle. **`--files-from`
hands it the list → zero scanning**, so retries are nearly free.

### 2.5 `tar -z` on `.nii.gz` / `.npz` is pure waste; `tar -h` doubles your upload
Measured re-gzip ratio on a `.nii.gz`: **1.000** (already compressed). And `tar -h` **dereferences hardlinks**
— it turned 430 unique images into 846 copies, inflating 12 GB → ~25 GB.

### 2.6 macOS openrsync rejects modern flags
No `--info=progress2` (GNU rsync 3.x only) → use `--progress`. Also rejects `--append-verify`.
Supported and useful: `--partial`, `--progress`, `--files-from`, `--timeout`, `-rt`.

### 2.7 The tail crawls because parallelism COLLAPSES onto the last files (E7, 2026-07-18)
Symptom: flies through the first ~90% of GB, then the last ~100 files crawl for an hour with intermittent
`server not responding` / `unexpected end of file`. Cause — **two compounding traps:**
- **Parallelism collapse.** Re-running a mostly-done chunk list lets streams whose files are all present
  **finish and exit**, dumping the remaining files onto the 2–3 streams that still hold them → ⅓ the throughput.
  Also, `sort` + `split -l` puts all the big same-directory files (e.g. `imagesTr/`) **contiguous → into 2–3
  chunks**, so they were never spread across 8 streams to begin with. **Fix: chunk ROUND-ROBIN and over only the
  EXACT missing files.** Compute the gap (`comm -23` of local vs `ssh pod 'ls dest'`), then
  `awk '{print > ("/tmp/m_" (NR%8))}'` so every stream carries real work start-to-finish.
- **Connection saturation.** 8 rsync streams + a Mac-side `du`/count monitor reconnecting every 20 s + 2 SSH
  sessions can trip the pod sshd's `MaxStartups` (default 10) → new connections refused, streams die, retries
  storm. **Fix: raise it on the pod** (`echo 'MaxStartups 100:30:200' >> /etc/ssh/sshd_config; pkill -HUP sshd`
  — HUP reloads without dropping sessions) and **monitor FROM the pod** (`while true; do du -sh /workspace;
  sleep 20; done`), never via a Mac reconnect loop.
- **Self-healing streams.** Wrap each stream in `until rsync ...; do sleep 3; done` (not `for try…&& break`) so
  it retries until it truly exits 0 — guaranteed convergence on a flaky link.
- **Don't download the `--npz` validation dumps.** An nnU-Net results dir is ~4.4 GB/expert, almost all
  disposable validation softmax. Pull only `fold_0/checkpoint_final.pth` (~235 MB) + `plans.json` +
  `dataset.json` + `dataset_fingerprint.json` to reload the model. Saved ~8.5 GB of pointless download on E7.

---

## 3. THE RECIPE (copy-paste)

```bash
export POD=root@<HOST>; export PORT=<PORT>; export KEY=~/.ssh/id_ed25519

cd <source-root>                       # paths in the list are relative to THIS dir
find <subdir> -type f > /tmp/files.txt # or: ls imagesTr/*.nii.gz labelsTr/*.nii.gz > /tmp/files.txt
sort -o /tmp/files.txt /tmp/files.txt
total=$(wc -l < /tmp/files.txt); per=$(( (total + 7) / 8 )); rm -f /tmp/chunk_*
split -l $per /tmp/files.txt /tmp/chunk_

for c in /tmp/chunk_*; do ( for try in $(seq 1 8); do \
  rsync -rt --partial --timeout=120 --files-from="$c" \
    -e "ssh -p $PORT -i $KEY -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes" \
    . "$POD:<dest>/" && break; echo "[$c] stalled, retry $try"; sleep 5; done ) & done; wait
echo "ALL CHUNKS DONE"
```

Notes:
- `--files-from` implies `--relative`, so **directory structure is recreated automatically** — no `mkdir` of
  subdirs needed (the dest root must exist).
- `ServerAliveInterval=15` + `--timeout=120` kill a stalled socket in ~1 min so the retry can resume from
  `--partial`. This is what defeats the **multi-minute freezes**.
- Ctrl-C won't stop it (background subshells own their process groups) → `pkill -f chunk_` if needed.
- **One huge file** → `split -b 2000M big.h5 big.h5.part_`, upload the parts with the recipe above, then on
  the pod: `for p in $(ls parts/*.part_* | sort); do cat "$p" >> big.h5 && rm "$p"; done`
  **Then verify `stat -c%s` against the source's exact byte count AND a content check** — size alone won't
  catch parts concatenated out of order.

### Rebuild duplicates on the pod instead of uploading them
If two datasets share identical files (we built Dataset003/004 to reuse Dataset002's **exact case IDs**
precisely so this would work):
```bash
for f in $D2/imagesTr/*.nii.gz; do ln -f "$f" $D4/imagesTr/$(basename $f); done
```
Instant, free, and byte-exact. **Design your datasets to make this possible** — it saved ~13 GB of upload.

---

## 4. VERIFY BEFORE YOU TRUST (never skip)

```bash
stat -c%s file                                    # exact bytes, both sides
ls dir/*.nii.gz | wc -l                           # case counts
python3 -c "import h5py; ..."                     # h5: patients AND dataset count, not just size
python3 -c "import torch; torch.load(f, map_location='cpu')"   # checkpoints: load, don't just stat
```
**Size alone proves nothing.** A truncated h5 still opens; parts concatenated out of order still stat
correctly. Check **content counts** (we verified 667 patients / 34,415 slices / 10,623,088,306 bytes).
Before terminating a pod, **diff the file inventory** pod-vs-local (`find | wc -l` per dir) — we caught 2
missed files that way, one of which mattered.

---

## 5. GPU / COST LESSONS

| Lesson | Detail |
|---|---|
| **Parallelism beats a faster chip** | 3× L40S ($2.97/hr) ran 3 experiments at once — **faster AND cheaper** than 1× H100 SXM ($2.99/hr) sequentially. Check the **"max" GPU count** per type: RTX 4090 caps at 2/pod, L40S allowed 3. |
| **Measured epoch times** | nnU-Net `3d_fullres` (patch [48,224,192], b2): **H100 14.8 s** · **L40S 35.3 s** (~2.5× slower). nnU-Net `2d` (512², b12): L40S **26.4 s**. TransUNet 102M @ 512² b8 AMP: **~4 min/epoch** on L40S. |
| **The pod bills while you upload** | We wasted ~$3 paying $2.97/hr to watch a progress bar. **Pull finished artifacts DURING training** (network ≠ GPU). Consider uploading to a network volume from a $0.25/hr pod first, then attaching to the GPU box. |
| **Disk: "80 GB total" = 50 GB volume + 30 GB overlay** | Not enough. Set **Volume Disk explicitly** (we used 400 GB). Disk is ~$0.0001/GB/hr — **effectively free, always oversize it.** |
| **nnU-Net preprocessed ≈ 190 MB/case** | float32 [123,512,512] + seg + the `.npz` (unpack keeps **both** `.npz` and `.npy`). 208 cases ≈ 40 GB; 430 ≈ 82 GB. **Cropping does NOT help here** — CT background is air at raw pixel ~7192, not 0, so `crop_to_nonzero` trims nothing. |
| **Restarting a pod wipes pip and changes the SSH port** | Re-`pip install` every time; get the new port from Connect. `/workspace` (the volume) survives **Stop**; **Terminate** destroys the pod (network volumes persist separately). |
| **Stop ≠ Terminate** | Stop keeps the volume at ~$0.01/hr — use it when pausing. |
| **macOS junk breaks globs** | `find /workspace -name '._*' -delete` — `._*` files ride in with uploads and break nibabel/glob steps. |

---

## 6. RUNNING JOBS

- **Always tmux.** `Ctrl-b c` new window · `Ctrl-b d` detach · `tmux attach -t <name>`.
  **The window showing live training output is READ-ONLY — Ctrl-C there kills the run.** Open a new window.
- **Chain sequential runs with `;`** so the next starts unattended: `train A ... ; train B ...`
- **Parallel multi-GPU:** `CUDA_VISIBLE_DEVICES=N cmd &` per job, then `wait`. Redirect each to its own log —
  **the parent then prints NOTHING until all finish, which looks like a hang but isn't.** Watch the logs.
- **NEVER re-run a script that auto-resumes training while training is live.** nnU-Net resumes from
  `checkpoint_latest.pth` with `--c`; a second invocation collides with the running job and corrupts both.
  Guard on `checkpoint_final.pth`; launch extra runs manually.
- **Make scripts skip incomplete inputs rather than die, and make them resumable.** Pods drop, balances run
  out. This let us start training on Dataset002 while Dataset003 was still uploading.

---

## 7. WHAT THIS COST / SAVED

- Track A: 3 experiments, 3× L40S, **~$10 including upload**.
- Track B: 4 experiments, 1–2× L40S, **~$12**.
- The chunked-parallel fix turned an **~8 h upload into ~45 min** — and unblocked a run that was otherwise
  going to die from an exhausted balance.

---

*Reference implementations: `ablation_study/RUNPOD_COMMANDS_TrackA.md` (§2 has the corrected upload),
`ablation_study/pod_run_all_trackA.sh` (preflight asserts, resumable, parallel dispatch, silent-failure
canaries).*
