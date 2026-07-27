#!/usr/bin/env bash
# =============================================================================
# Track A — run EVERYTHING that needs a GPU, end to end, in one go.
#   E1  = nnU-Net 2d          on Dataset002_Parotid       (2D vs 3D)
#   E2  = nnU-Net 3d_fullres  on Dataset003_ParotidDirty  (annotation gap, dirty-430)
#   E2b = nnU-Net 3d_fullres  on Dataset004_ParotidGapped (annotation gap, constant-N 208)
#
# For each: preprocess -> train (fold 0, no-mirror) -> predict (--disable_tta)
#           -> evaluate against the LOCKED clean Dataset002 test labels.
#
# RESUMABLE: safe to re-run after a pod restart / disconnect. Finished stages are
# skipped; an interrupted training resumes from its latest checkpoint (--c).
#
# USAGE (on the pod, inside tmux):
#   tmux new -s trackA
#   bash /workspace/pod_run_all_trackA.sh 2>&1 | tee /workspace/trackA_run.log
#   # detach: Ctrl-b d     reattach: tmux attach -t trackA
#
# Total expected: ~5-8 GPU-hours on a 4090 (~$3-6).
# =============================================================================
set -euo pipefail

WORK=${WORK:-/workspace}
TR=nnUNetTrainer_250epochs_noMirror
FOLD=0

export nnUNet_raw=$WORK/nnUNet_raw
export nnUNet_preprocessed=$WORK/nnUNet_preprocessed
export nnUNet_results=$WORK/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$WORK/preds" "$WORK/results"

GT_DIR="$nnUNet_raw/Dataset002_Parotid/labelsTs"   # the locked 43-case test set — never changes
MANIFEST="$WORK/results/run_manifest.txt"

# id | dataset folder | config | short name | eval model-name
#
# ORDER IS DELIBERATE — highest scientific value first, so that if the pod dies or the balance runs
# out partway, the most important results are already banked:
#   1. E2b  -> answers Q2 (annotation gap) CLEANLY: constant N, same patients, only labels differ.
#   2. E1   -> answers Q1 (2D vs 3D). After these two, both headline questions are answered.
#   3. E2   -> the realistic dirty-430 arm; adds the "what did 222 extra patients buy back"
#              decomposition, but is confounded on its own (N doubles), so it goes last.
EXPERIMENTS=(
  "4|Dataset004_ParotidGapped|3d_fullres|E2b_gapped|nnUNet_3d_gapped208_fold0"
  "2|Dataset002_Parotid|2d|E1_2d|nnUNet_2d_fold0"
  "3|Dataset003_ParotidDirty|3d_fullres|E2_dirty|nnUNet_3d_dirtylabels_fold0"
)

log()  { echo -e "\n\033[1;36m==> $*\033[0m"; }
ok()   { echo -e "\033[1;32m  [OK] $*\033[0m"; }
warn() { echo -e "\033[1;33m  [!!] $*\033[0m"; }
die()  { echo -e "\033[1;31m  [XX] $*\033[0m"; exit 1; }

# ---------------------------------------------------------------- 0. preflight
log "STEP 0 — preflight"

python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'" \
  || die "No CUDA GPU. Check the pod has a GPU attached."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME"

for dep in nnunetv2 nibabel scipy skimage; do
  python3 -c "import $dep" 2>/dev/null || { warn "installing deps..."; pip install -q nnunetv2 nibabel scipy scikit-image; break; }
done
ok "deps present (nnunetv2 $(python3 -c 'import nnunetv2;print(getattr(nnunetv2,"__version__","?"))'))"

# Expected case counts per dataset. A dataset that is absent or still uploading is SKIPPED (with a
# warning) rather than fatal — so you can start training on whatever has fully landed and re-run this
# script later to pick up the rest. A dataset that is present but WRONG is still fatal.
declare -A EXPECT_TR=( [Dataset002_Parotid]=208 [Dataset003_ParotidDirty]=430 [Dataset004_ParotidGapped]=208 )
READY=()
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ ds _ name _ <<< "$e"
  n_tr=$(ls "$nnUNet_raw/$ds/labelsTr"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ')
  n_ti=$(ls "$nnUNet_raw/$ds/imagesTr"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ')
  n_ts=$(ls "$nnUNet_raw/$ds/labelsTs"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ')
  n_is=$(ls "$nnUNet_raw/$ds/imagesTs"/*.nii.gz 2>/dev/null | wc -l | tr -d ' ')
  exp=${EXPECT_TR[$ds]}
  if [ ! -d "$nnUNet_raw/$ds" ] || [ "$n_tr" -eq 0 ]; then
    warn "$ds: ABSENT -> skipping $name (re-run this script once it has uploaded)"
    continue
  fi
  if [ "$n_tr" -ne "$exp" ] || [ "$n_ti" -ne "$exp" ] || [ "$n_ts" -ne 43 ] || [ "$n_is" -ne 43 ]; then
    warn "$ds: INCOMPLETE (imagesTr=$n_ti labelsTr=$n_tr imagesTs=$n_is labelsTs=$n_ts; expected $exp/$exp/43/43)"
    warn "  -> skipping $name; finish the upload and re-run this script."
    continue
  fi
  ok "$ds: $n_tr train / $n_ts test — complete"
  READY+=("$e")
done
[ "${#READY[@]}" -gt 0 ] || die "no complete dataset found — nothing to do yet"
EXPERIMENTS=("${READY[@]}")
ok "${#EXPERIMENTS[@]} experiment(s) ready to run this pass"

{ echo "Track A run manifest"; echo "date       : $(date -u +%Y-%m-%dT%H:%M:%SZ)";
  echo "GPU        : $GPU_NAME"; echo "trainer    : $TR (mirroring disabled, 250 epochs)";
  echo "fold       : $FOLD"; echo "test set   : $GT_DIR (43 locked cases)";
  echo "seed       : nnU-Net default (deterministic per fold; see fold_0/debug.json)"; } > "$MANIFEST"

# ------------------------------------------------- 1. register the custom trainer
log "STEP 1 — register the no-mirror trainer"
NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
[ -f "$WORK/nnUNetTrainer_250epochs_noMirror.py" ] || die "missing $WORK/nnUNetTrainer_250epochs_noMirror.py (scp it from the Mac)"
cp "$WORK/nnUNetTrainer_250epochs_noMirror.py" "$NNUNET_DIR/training/nnUNetTrainer/variants/"
python3 - <<'PY' || die "trainer did NOT register — training would silently fall back / fail"
import os, nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
c = recursive_find_python_class(
    os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer'),
    'nnUNetTrainer_250epochs_noMirror', 'nnunetv2.training.nnUNetTrainer')
assert c is not None, "not found"
print(f"  [OK] trainer registered: {c}")
PY

# ------------------------------------------------------------- 2. preprocessing
log "STEP 2 — plan & preprocess"
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r id ds cfg name _ <<< "$e"
  if [ -d "$nnUNet_preprocessed/$ds/${cfg}" ] || [ -d "$nnUNet_preprocessed/$ds/nnUNetPlans_${cfg}" ]; then
    ok "$name: already preprocessed, skipping"
  else
    echo "  preprocessing $ds ($cfg) ..."
    nnUNetv2_plan_and_preprocess -d "$id" -c "$cfg" --verify_dataset_integrity
    ok "$name preprocessed"
  fi
done

log "STEP 2b — plans check (comparability)"
# If nnU-Net picks a DIFFERENT patch/spacing for Dataset003/004 than Dataset002, the runs are not
# purely a label comparison. Report it; do NOT silently 'fix' it.
python3 - <<'PY' | tee -a "$MANIFEST"
import json, os
pre = os.environ['nnUNet_preprocessed']
for ds in ['Dataset002_Parotid', 'Dataset003_ParotidDirty', 'Dataset004_ParotidGapped']:
    p = os.path.join(pre, ds, 'nnUNetPlans.json')
    if not os.path.exists(p):
        print(f"  {ds}: plans.json not found"); continue
    cfgs = json.load(open(p))['configurations']
    for cfg in ('2d', '3d_fullres'):
        if cfg in cfgs:
            c = cfgs[cfg]
            sp = [round(float(x), 3) for x in c.get('spacing', [])]
            print(f"  {ds:28s} {cfg:11s} spacing={sp} patch={c['patch_size']} batch={c['batch_size']}")
PY
warn "Dataset003/004 3d_fullres should match Dataset002's spacing [3.0, 0.977, 0.977] & patch [48, 224, 192]."
warn "If they differ, note it in RESULT.md — it means nnU-Net re-planned, adding a second variable."

# ------------------------------------------------------------------ 3. training
NGPU=$(nvidia-smi -L | wc -l)
log "STEP 3 — training ($NGPU GPU(s) visible)"
echo "  reference: this exact 3d_fullres config ran at 14.8 s/epoch x 250 epochs = ~66 min on an H100 80GB."

# Build the list of runs that still need training.
TODO=()
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r id ds cfg name _ <<< "$e"
  outdir="$nnUNet_results/$ds/${TR}__nnUNetPlans__${cfg}/fold_${FOLD}"
  if [ -f "$outdir/checkpoint_final.pth" ]; then
    ok "$name: checkpoint_final.pth exists — already trained, skipping"
  else
    TODO+=("$e")
  fi
done

train_one() {  # $1 = experiment spec, $2 = gpu index ("" = whatever is visible)
  IFS='|' read -r id ds cfg name _ <<< "$1"
  local outdir="$nnUNet_results/$ds/${TR}__nnUNetPlans__${cfg}/fold_${FOLD}"
  local resume=""
  [ -f "$outdir/checkpoint_latest.pth" ] && resume="--c"
  if [ -n "$2" ]; then
    CUDA_VISIBLE_DEVICES="$2" nnUNetv2_train "$id" "$cfg" "$FOLD" -tr "$TR" --npz $resume
  else
    nnUNetv2_train "$id" "$cfg" "$FOLD" -tr "$TR" --npz $resume
  fi
}

t_all=$(date +%s)
if [ "${#TODO[@]}" -eq 0 ]; then
  ok "nothing to train"
elif [ "$NGPU" -ge "${#TODO[@]}" ] && [ "${PARALLEL:-1}" = "1" ]; then
  # ---- PARALLEL: one experiment per GPU, all at once (wall-clock ~= a single run) ----
  ok "PARALLEL mode: running ${#TODO[@]} trainings concurrently, one per GPU"
  pids=(); pnames=()
  gi=0
  for e in "${TODO[@]}"; do
    IFS='|' read -r _ _ _ name _ <<< "$e"
    echo "  launching $name on GPU $gi -> $WORK/results/train_${name}.log"
    train_one "$e" "$gi" > "$WORK/results/train_${name}.log" 2>&1 &
    pids+=("$!"); pnames+=("$name"); gi=$((gi+1))
  done
  echo "  waiting... (tail a log to watch, e.g. tail -f $WORK/results/train_${pnames[0]}.log)"
  fail=0
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      ok "${pnames[$idx]} finished"
    else
      warn "${pnames[$idx]} FAILED — see $WORK/results/train_${pnames[$idx]}.log"; fail=1
    fi
  done
  [ "$fail" -eq 0 ] || die "at least one parallel training failed; re-run this script to resume the rest"
else
  # ---- SEQUENTIAL fallback (fewer GPUs than runs, or PARALLEL=0) ----
  [ "$NGPU" -lt "${#TODO[@]}" ] && warn "only $NGPU GPU(s) for ${#TODO[@]} runs -> sequential (slower wall-clock)"
  for e in "${TODO[@]}"; do
    IFS='|' read -r _ _ _ name _ <<< "$e"
    t0=$(date +%s)
    echo "  training $name ..."
    train_one "$e" "" 2>&1 | tee "$WORK/results/train_${name}.log"
    ok "$name trained in $(( ($(date +%s)-t0)/60 )) min"
  done
fi
echo "walltime   : all training = $(( ($(date +%s)-t_all)/60 )) min on ${NGPU}x $GPU_NAME" >> "$MANIFEST"
ok "training stage done in $(( ($(date +%s)-t_all)/60 )) min"

# ---------------------------------------------------------------- 4. prediction
log "STEP 4 — predict the locked test set (mirroring OFF)"
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r id ds cfg name _ <<< "$e"
  pdir="$WORK/preds/$name"
  if [ "$(ls "$pdir"/*.nii.gz 2>/dev/null | wc -l)" -eq 43 ]; then
    ok "$name: 43 predictions already present, skipping"
    continue
  fi
  mkdir -p "$pdir"
  # each dataset's imagesTs is a byte-identical clone of Dataset002's -> same 43 cases everywhere
  nnUNetv2_predict -i "$nnUNet_raw/$ds/imagesTs" -o "$pdir" \
    -d "$id" -c "$cfg" -tr "$TR" -f "$FOLD" --disable_tta
  n=$(ls "$pdir"/*.nii.gz 2>/dev/null | wc -l)
  [ "$n" -eq 43 ] || die "$name produced $n predictions, expected 43"
  ok "$name: 43 predictions written"
done

# ---------------------------------------------------------------- 5. evaluation
log "STEP 5 — evaluate (always against the CLEAN Dataset002 test labels)"
[ -f "$WORK/eval_testset.py" ] || die "missing $WORK/eval_testset.py (scp it from the Mac)"
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r id ds cfg name mname <<< "$e"
  csv="$WORK/results/eval_${name}.csv"
  echo "  evaluating $name ..."
  python3 "$WORK/eval_testset.py" \
    --pred-dir "$WORK/preds/$name" \
    --gt-dir   "$GT_DIR" \
    --model-name "$mname" \
    --results-csv "$csv"
  ok "$name -> $csv"
done

# ------------------------------------------------------------------- 6. summary
log "STEP 6 — SUMMARY (send these numbers back to the agent)"
echo
echo "  Reference (already known, from the main project):"
echo "    nnU-Net 3d_fullres clean, 1 fold : Dice 0.8187 | HD95 5.25 | SurfDice 0.8965"
echo "    nnU-Net 5-fold ensemble          : Dice 0.8202 | HD95 5.24   (E5: delta n.s., p=0.35)"
echo
echo "  This run:"
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ _ _ name _ <<< "$e"
  csv="$WORK/results/eval_${name}.csv"
  [ -f "$csv" ] && { echo "  --- $name ---"; cat "$csv"; echo; }
done | tee -a "$MANIFEST"

# internal val Dice — the mirror-bug canary (noMirror should be ~0.79-0.80; ~0.45 means mirroring is back)
echo "  Internal fold-0 validation Dice (sanity — expect ~0.79-0.80, NOT ~0.45):"
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r _ ds cfg name _ <<< "$e"
  logf=$(ls "$nnUNet_results/$ds/${TR}__nnUNetPlans__${cfg}/fold_${FOLD}"/training_log_*.txt 2>/dev/null | tail -1 || true)
  [ -n "$logf" ] && echo "    $name: $(grep -a 'Pseudo dice' "$logf" | tail -1 | sed 's/^.*Pseudo dice/Pseudo dice/')"
done | tee -a "$MANIFEST"

echo
ok "ALL DONE. Manifest: $MANIFEST"
echo
echo "  Pull everything back to the Mac with (run ON THE MAC):"
echo "    cd \"/Users/ritvikmod/Desktop/Academics/Projects/ct scan project\""
echo "    scp -P \$PORT -r \$POD:$WORK/results ablation_study/_pod_results"
echo
echo "  Then tell the agent 'pod results are in ablation_study/_pod_results' and it will"
echo "  fill in E1/E2/E2b RESULT.md, the status table, and master section 20.A."
