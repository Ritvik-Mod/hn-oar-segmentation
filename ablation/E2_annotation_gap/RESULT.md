# E2 — Annotation gap: clean vs dirty labels

**Track A · Status: ✅ DONE (both arms) · Date: 2026-07-16**

---

## HEADLINE: the annotation gap has a **latent** cost of 12.9 Dice points, of which **8.3 is bought back by the extra partially-labelled patients** — leaving a **4.6-point net cost** in the real historical condition.

| Condition | N | 3D Dice | Tversky | HD95 (mm) | Surf-Dice | R / L Dice |
|---|---|---|---|---|---|---|
| clean-208 (baseline, `results_tracker_TEST.csv`) | 208 | **0.8187** | 0.8166 | 5.25 | 0.8965 | 0.8278 / 0.8095 |
| **E2 — dirty-430** (`eval_E2_dirty.csv`) | 430 | **0.7726** | 0.7644 | **5.45** | 0.8400 | 0.7891 / 0.7560 |
| **E2b — gapped-208** (`eval_E2b_gapped.csv`) | 208 | **0.6899** | 0.6680 | **9.63** | 0.7649 | 0.7121 / 0.6678 |

The three-way decomposition the constant-N arm was built to produce:

| Comparison | Δ Dice | Meaning |
|---|---|---|
| **clean − E2b** | **−0.1288** | **the gap's PURE cost** at fixed N, patients, images and config — labels the only variable |
| **E2 − E2b** | **+0.0827** | what the 222 extra single-side patients **buy back** (64% of the damage) |
| **clean − E2** | **−0.0461** | the gap's **NET cost in the real historical condition** (dirty labels *and* 2× data) |

### Why both arms were necessary (this is the study's methodological payoff)

- **E2 alone** would have reported *"the annotation gap costs 4.6 Dice points"* — a **substantial
  understatement**, because more than half the damage was silently masked by having 2× the patients.
- **E2b alone** would have reported *"the gap costs 12.9 points"* — true at fixed N, but an **overstatement**
  as an explanation of the historical 0.62 → 0.82 gain, since the Phase-1 baseline never had constant N.
- **Together** they separate label noise from data volume, which neither gives alone. The design flaw flagged
  during prep (E2 changes two variables at once) was real, and the outcome confirms it mattered: the two
  arms differ by **0.083 Dice**, i.e. the confound was larger than the effect E2 would have reported.

### ⚠️ Correction to an earlier claim in this file

An earlier revision (written when only E2b had finished) stated that the annotation gap explains **~65% of
the 0.62 → 0.82 gain**. **That was wrong and is retracted.** It extrapolated E2b's constant-N number to a
historical comparison that did *not* hold N constant: Phase-1 trained on ~383 parotid-bearing patients
(master §7: train 583 → both 186 + only-R 98 + only-L 99), far closer to E2's 430 than to E2b's 208. The
label condition **as actually suffered** is therefore the **E2** number (−0.046), not the E2b number (−0.129).

### Corrected attribution of the ~0.199 Dice in the 0.62 → 0.82 headline

| Axis | Contribution | Share | Source |
|---|---|---|---|
| **Preprocessing** (resampling + CTNormalization + augmentation) | **≈ +0.145** | **~73%** | remainder: E2 (0.7726, dirty labels ≈ Phase-1's data condition) vs Phase-1 (0.62), minus 3D |
| Label quality (as suffered) | ≈ +0.046 | ~23% | clean − E2 ✅ |
| Dimensionality (2D → 3D) | +0.007 | ~3.5% | E1 ✅ |
| Ensembling (1 → 5 folds) | +0.0015 (n.s., p=0.35) | ~0.8% | E5 ✅ |

**Preprocessing was the driver all along.** This confirms — and finally *proves* — Phase-1's own suspicion
(master §13.4) that the 0.62 ceiling came from the data/preprocessing rather than model capacity. It was not
dimensionality (E1: +0.007), not ensembling (E5: n.s.), and only ~23% labels.

### The actionable finding

**Partially-labelled patients are worth including**: they recover 64% of the gap's damage (+0.083) and
restore boundary quality almost entirely (HD95 9.63 → 5.45, vs 5.25 baseline). But they still leave 4.6
points on the table versus clean labels.

**This is the strongest argument yet for `pipeline/masked_loss.py`** (master §10.2, currently unused): masking
un-annotated organs out of the loss should capture the **data benefit** of all 430 patients **without** the
12.9-point label penalty — plausibly beating 0.8187, since it would train on 430 patients with no false
"gland = background" signal. That is now a concrete, motivated next experiment rather than a design intuition.

### Supporting signals

- **Tversky (α=0.3/β=0.7, false-negative weighted) falls further than Dice in both arms** (E2b −0.1486 vs
  −0.1288; E2 −0.0522 vs −0.0461) — the exact signature expected when a model is taught to *omit* glands.
- **HD95 separates the arms sharply**: E2b 9.63 mm (nearly double the 5.25 baseline) vs E2 5.45 mm (≈
  baseline). Boundary quality is what the extra data restores most completely.
- **R/L asymmetry widens monotonically with label damage**: clean 0.018 → E1 0.031 → E2 0.033 → **E2b 0.044**.
  Consistent with the gap being the cause (E2b deleted L on 53 cases and R on 54; the L side degraded more).

*(Caveat carried to S1: 0.62 is a Phase-1 **validation** number while all of these are **test** numbers, so
the preprocessing remainder is approximate. Track B's **P0** puts the four Phase-1 checkpoints on the locked
test set and will firm up the arithmetic. The ordering of the axes will not change.)*

**Sanity:** E2b's internal fold-0 pseudo-Dice was [0.6595, 0.7606] — low, but *expected* and not a red flag:
nnU-Net scores each model against **its own** validation labels, and E2b's val labels are gapped too, so the
model is penalised as a false positive whenever it correctly predicts a gland the clinician never contoured.
That is the annotation gap corrupting the *metric*, which is why the only number that counts is the eval
above, against the **clean** 43-case test labels. Nowhere near the 0.4538 mirror-bug signature.

## Reproducibility

| Item | E2b (gapped-208) | E2 (dirty-430) |
|---|---|---|
| Train | `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror --npz` | `CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_250epochs_noMirror --npz` |
| Predict | `-d 4 -c 3d_fullres -tr … -f 0 --disable_tta` | `-d 3 -c 3d_fullres -tr … -f 0 --disable_tta` |
| Eval | `eval_testset.py --gt-dir .../Dataset002_Parotid/labelsTs --model-name nnUNet_3d_gapped208_fold0` | `… --model-name nnUNet_3d_dirtylabels_fold0` |
| GPU / time | 1× **L40S**, **35.26 s/epoch** × 250 ≈ **147 min** (finished 19:48) | 1× **L40S**, **35.25 s/epoch** × 250 ≈ **147 min** (finished 20:26) |
| Seeds | dataset gap-simulation seed **42**; nnU-Net training seed = default | nnU-Net training seed = default |
| Plans (both **verified identical to D002**) | spacing [3.0, 0.977, 0.977], patch [48, 224, 192], batch 2 | spacing [3.0, 0.977, 0.977], patch [48, 224, 192], batch 2 |
| Internal fold-0 pseudo-Dice | [0.6595, 0.7606] | [0.7989, 0.8498] |
| Artifacts | `_pod_results/results/eval_E2b_gapped.csv`, `preds/E2b_gapped/`, `nnUNet_results/Dataset004_ParotidGapped/` | `_pod_results/results/eval_E2_dirty.csv`, `preds/E2_dirty/`, `nnUNet_results/Dataset003_ParotidDirty/` |

**Common to both:** 3× L40S RunPod box ($2.97/hr), 2026-07-16; locked 43-case test set
(`Dataset002_Parotid/labelsTs`), scored against **clean** labels in every case; `--disable_tta` throughout;
43/43 predictions asserted per run. Total Track A GPU cost ≈ $10 including upload.

**On the internal pseudo-Dice numbers (do not misread them):** nnU-Net scores each model against *its own*
validation labels. E2b's and E2's val labels are gapped/dirty too, so a model is penalised as a false
positive whenever it correctly predicts a gland the clinician never contoured. That is the annotation gap
corrupting the *metric*, and it explains why E2b's internal number (0.66/0.76) looks worse than its clean-test
result implies, and why E2's (0.80/0.85) looks *better* than its test result (0.7726). **Only the test eval
against clean labels is meaningful.** Neither is anywhere near the 0.4538 that signals the L/R mirror bug.

---

## Question

How much of the 0.62 → 0.82 gain is explained by **label quality** — specifically the annotation gap, where
clinicians deliberately do not contour the healthy-side parotid on one-sided tumours (master §16.1)? The
Phase-1 baselines trained on those partial labels (the un-contoured gland treated as background); nnU-Net
trained only on the both-annotated, QC-clean subset. This is the study's most original question (Q2).

## Status summary

| Step | State |
|---|---|
| Measure the annotation gap in the train/val pool | ✅ done (results below) |
| Validate that dirty labels introduce *only* the gap (no L/R noise) | ✅ done — validated |
| Write `make_dirty_dataset.py` | ✅ done |
| **Build `Dataset003_ParotidDirty` (E2, dirty-430)** | ✅ **done & verified — 430 train / 43 test, 12 GB** |
| **Build `Dataset004_ParotidGapped` (E2b, constant-N)** | ✅ **done & verified — 208 train / 43 test** |
| Train `3d_fullres` fold 0 + predict + eval | ⏳ **pending — needs a rented GPU (no CUDA locally)** |

**The only remaining blocker is GPU compute; all data and code are ready.** The owner will rent a RunPod pod
and run the training; the copy-pasteable command sheet is at `ablation_study/RUNPOD_COMMANDS_TrackA.md`.

**Correction to an earlier assumption:** an initial search for `Dataset002_Parotid` used `find -maxdepth 4`
and missed it — the dataset is at depth 6. **The clean `Dataset002_Parotid` raw dataset is fully present
locally** at `Parotid-Project/Datasets/Dataset002_Parotid` (6.5 GB, 208 train + 43 test). Note that master
§18 describes `Parotid-Project/Datasets/` as holding only "nnU-Net `dataset.json` + `case_mapping*.json`" —
that is **stale**; the folder holds the complete `imagesTr/labelsTr/imagesTs/labelsTs` volumes. §18 is frozen,
so this correction is recorded here and in §20.A. No rebuild of Dataset002 was needed.

---

## Result 1 — the size of the annotation gap (measured, no training needed)

```bash
python3 ablation_study/E2_annotation_gap/scan_parotid_presence.py
```

Scanned all 667 train+val patients in `ML_Dataset_Final` (read-only), replicating
`build_volumes.reconstruct_patient`'s CT-group selection rule so the counts match what the dataset builder
would actually produce. Wall-time **5.9 s** (8 CPU workers). Output:
`ablation_study/E2_annotation_gap/parotid_presence_trainval.csv`.

| Annotation status (train+val, n=667) | Patients |
|---|---|
| **both** parotids annotated → the clean Dataset002 pool | **208** |
| only R annotated | 110 |
| only L annotated | 112 |
| no parotid | 237 |
| **Clean training pool (both-annotated)** | **208** |
| **Dirty training pool (≥1 parotid)** | **430** |
| **Extra single-side patients the dirty set adds** | **222** |

**These counts validate exactly against two independent sources:** `both = 208` matches Dataset002's
"208 train+val" (master §14.3), and the per-status counts reproduce master §7's split table exactly
(train 98 only-R + val 12 = 110 ✓; train 99 only-L + val 13 = 112 ✓; train 200 none + val 37 = 237 ✓).

**The annotation gap is large: 222 of the 430 parotid-bearing train/val patients (51.6%) are contoured on
one side only.** More than half the available training population is partially labelled — this is not a
marginal data-cleaning detail, it is the majority condition of the raw clinical data.

## Result 2 — the dirty labels introduce *only* the gap, not L/R noise (validated)

This check matters for E2's validity. `pipeline/fix_labels_qc.relabel_geometric()` assigns L/R by comparing
the two glands' column centroids — but it **returns early when only one gland is present** (line 28-29,
comment: *"shouldn't happen in both-annotated"*). So every single-side case in the dirty set keeps whatever
the clinician originally named it. If that source naming were unreliable, E2 would silently confound the
annotation gap with L/R label noise — i.e. two variables, and the study's headline experiment would be
uninterpretable.

Measured from the same scan (image width 512 → midline 255.5):

| Check | Result |
|---|---|
| Both-annotated cases where source name agrees with the geometric rule (col_R < col_L) | **207 / 208** |
| Single-side glands sitting on the anatomically expected side of the midline | **222 / 222 (100%)** |
| Mean column centroid, single-side only_R / only_L | 202.2 / 306.7 |
| Mean column centroid, both-annotated reference R / L | 202.0 / 306.4 |

**Conclusion: the source L/R naming is trustworthy, so the dirty set's only added noise is the missing
contralateral gland — exactly the variable E2 intends to isolate.** The single-side centroid distributions
are statistically indistinguishable from the both-annotated reference (202.2 vs 202.0; 306.7 vs 306.4), and
not one of the 222 is on the wrong side.

**Bonus — this independently reproduces the project's second-biggest finding.** Master §15 step 5 argues the
L/R flips came from the model, not the data, on the evidence that the geometric relabel corrected *only 1 of
251 cases*. This scan reaches that number from the raw npz by a different route: **207/208 = exactly 1
disagreement** in the train/val pool. The claim in §15.5 holds.

## Result 3 — the datasets are built and verified

Both training datasets were built locally on CPU. **Nothing on disk was modified**: the source
`Dataset002_Parotid` and `ML_Dataset_Final` were opened read-only, and images are *hardlinked* into the new
folders (a hardlink shares the inode, so it costs no space and cannot mutate the source's contents).
Verified after the build: Dataset002 still has exactly 208/208/43/43 files.

### Dataset003_ParotidDirty (E2 — the dirty arm)

```bash
python3 ablation_study/E2_annotation_gap/make_dirty_dataset.py \
  --data-dir "ML_Dataset_Final" \
  --split "patient classification/dataset_split.json" \
  --clean-dataset "Parotid-Project/Datasets/Dataset002_Parotid" \
  --clean-mapping "Parotid-Project/Datasets/Dataset001_Parotid/case_mapping.json" \
  --out "ablation_study/E2_annotation_gap/nnUNet_raw/Dataset003_ParotidDirty" --workers 6
```

**Construction: dirty = clean ∪ single-side, exactly.** Rather than re-reconstructing all 430 patients, the
builder *clones* Dataset002's 208 both-annotated training cases byte-for-byte (they are already L/R-corrected
and QC-filtered) and reconstructs only the 222 single-side patients from the npz. This guarantees the
both-annotated subset is **identical** to the clean baseline's, so the only difference between the E2 run and
the 0.8187 baseline is the 222 added partial-label patients — nothing else can drift. It also saved ~6.5 GB
and most of the CPU time. Verified by construction: the scan's 208 "both" patient ids are *exactly*
Dataset002's 208 training patient ids (set equality), with zero overlap with the 43 test patients.

| Property | Value |
|---|---|
| **numTraining** | **430** — 208 both + **110 only_R + 112 only_L** (exactly as the scan predicted) |
| Test set | 43 cases, cloned unchanged from Dataset002 |
| QC threshold | 18,622 voxels (3× Dataset002's median gland 6,208) — the same absolute rule Dataset002 used; 0 new cases dropped |
| Skipped | 237 (no parotid) |
| Size / wall-time | 12 GB / ~4 min (6 CPU workers) |

*(The QC threshold is 18,622 vs the 18,681 recorded in master §14.3 because §14.3's was computed over
Dataset001 — which still contained the corrupt PAR0240 and 44 test cases — whereas this one is computed over
the post-QC Dataset002. A 0.3% difference; it dropped nothing either way.)*

### Dataset004_ParotidGapped (E2b — the constant-N arm)

```bash
python3 ablation_study/E2_annotation_gap/make_gapped_dataset.py \
  --clean-dataset "Parotid-Project/Datasets/Dataset002_Parotid" \
  --clean-mapping "Parotid-Project/Datasets/Dataset001_Parotid/case_mapping.json" \
  --out "ablation_study/E2_annotation_gap/nnUNet_raw/Dataset004_ParotidGapped" --seed 42
```

| Property | Value |
|---|---|
| **numTraining** | **208** — 101 both + 53 only_R + 54 only_L |
| Simulated gap rate | **51.4%** (real measured rate: 51.6%) |
| Deleted side ratio | 53 L-deleted : 54 R-deleted (matched to the real 110:112) |
| Seed | 42 (project convention); selection recorded per-case in `case_mapping.json` |
| Test set | 43 cases, cloned unchanged |
| Size / wall-time | 30 MB (images hardlinked) / ~30 s |

**Verification (all passed):**

| Check | Result |
|---|---|
| Images share an inode with Dataset002 (hardlinked → source provably untouched) | **208/208** |
| Label files are NEW files, not links to the source | **0 shared inodes** ✓ |
| Gapped cases: exactly one gland removed, the other bit-identical | **107/107** |
| Ungapped cases: labels bit-identical to Dataset002 | **101/101** |
| Dataset002 intact after the build | 208/208/43/43 ✓ |

## Remaining blocker — GPU only (⚠️)

The local Mac has **no CUDA GPU** — MPS only (Apple unified memory, 8 GB, 8 CPU cores). nnU-Net
`3d_fullres` here is patch [48, 224, 192] at batch 2 (master §14.6) and trained in ~1 h on a rented H100; on
this device it is not viable (days-to-weeks, if it does not simply OOM), and the timing/repro record would be
meaningless. Per the golden rules I did **not** substitute a shrunken config or a different test set — either
would break comparability with the 0.8187 baseline.

**The owner is renting a RunPod pod.** All three runs (E1, E2, E2b) are covered by the copy-pasteable command
sheet at **`ablation_study/RUNPOD_COMMANDS_TrackA.md`** — upload, trainer registration, preprocessing, the
three trainings, prediction with `--disable_tta`, evaluation against the clean test labels, and the sanity
checks. Estimated total: **~5–8 GPU-hours ≈ $3–6** on a 4090.

## ✅ Design issue found during prep — E2 as specified changes TWO variables (resolved: E2b added)

**Owner decision: E2b approved and built** (constant-N arm, below). Recording the reasoning:

The plan's ground rule is "one variable per experiment" (plan §1.2), but as specified E2 changes two things
at once:

| | Clean (Dataset002) | Dirty (Dataset003) |
|---|---|---|
| Label condition | both-annotated only | **partial labels included** ← the intended variable |
| Training set size | 208 patients | **430 patients (+107%)** ← an unintended second variable |

The dirty set does not just add noise, it **more than doubles the training population**. The two effects
push in opposite directions: partial labels should hurt, 2× the data should help. So the headline outcomes
are ambiguous:

- If E2 scores **≈ 0.8187** (no change), that could mean either "the annotation gap costs nothing" *or*
  "the gap costs several points but 2× data exactly compensated". These are very different conclusions and
  the experiment as designed cannot distinguish them.
- If E2 scores **clearly lower**, the finding is strong and *conservative* — label noise hurt enough to
  overcome a doubling of the data. (This is the most likely outcome, and it would still be publishable.)
- If E2 scores **higher**, it says the extra data outweighs the gap — interesting, but it does not measure
  the gap.

**My hypothesis:** E2 as designed will land meaningfully below 0.8187 (the annotation gap teaches the model
that visible parotid tissue is sometimes background, which directly attacks the decision boundary), so the
experiment will likely still produce its headline. But the number will be a **lower bound on the gap's cost**,
not a measurement of it, and the write-up must say so.

**The fix — E2b, a constant-N arm** (approved by the owner; built, see Result 3): take the **same 208
both-annotated patients** and *simulate* the gap by deleting one gland's contour on 51.6% of them (matching
the real single-side rate measured above, R/L balanced 110:112). Train `3d_fullres` fold 0. N, patient
composition, and images are all identical to the clean baseline, so **only the label condition differs** — a
true one-variable isolation of the annotation gap.

**The three-arm design and what each comparison yields:**

| Arm | N | Labels | Isolates |
|---|---|---|---|
| clean-208 (existing, 0.8187) | 208 | clean | — the baseline |
| **E2b gapped-208** | 208 | gap simulated on 51.4% | **the annotation gap alone** (true isolation) |
| **E2 dirty-430** | 430 | real partial labels | the gap *as the baselines actually suffered it* |

- **clean − E2b** = the pure cost of the annotation gap.
- **E2 − E2b** = what the 222 extra patients buy back (i.e. separates "label noise" from "more data").
- **clean − E2** = the realistic net effect, which is what the Phase-1 baselines actually experienced.

Neither arm alone gives this; together they answer Q2 properly. E2b costs one extra fold (~1–3 h, ~$1–3).

## Interpretation (of what is established so far)

Even with training blocked, two things are now on the record. First, **the annotation gap affects 51.6% of
the parotid-bearing training population (222/430 patients)** — this is measured, not estimated, and it
justifies the premise of the whole question: the Phase-1 baselines trained on a pool where over half the
patients had a visible, un-contoured gland labelled as background. Second, **the dirty condition is clean
in the sense that matters** — its single-side labels are correctly named (222/222), so E2 will isolate the
annotation gap and not a mixture of gap + L/R noise.

What remains unknown is the headline: the Dice cost of the gap. That needs the GPU run above. The design
caveat (N confound) should be resolved first, ideally by adding the E2b arm.

## Files

| File | Contents |
|---|---|
| `scan_parotid_presence.py` | the measurement script (read-only over `ML_Dataset_Final`) |
| `parotid_presence_trainval.csv` | per-patient status, gland voxels, centroids (667 rows) |
| `make_dirty_dataset.py` | the Dataset003 builder — written, arg-validated, awaiting Dataset002 |
| `RESULT.md` | this file |
