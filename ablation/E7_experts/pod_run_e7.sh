#!/usr/bin/env bash
# E7 pod run — builds both experts, trains (parallel, 2 GPUs), predicts EVERYTHING
# (experts + the 4 Part-1 models), and quick-evals. Resumable & guarded (playbook §6).
# Expects the upload layout described in RUNPOD_COMMANDS_E7.md. Run inside tmux.
set -uo pipefail

WS=/workspace
export nnUNet_raw=$WS/nnunet/nnUNet_raw
export nnUNet_preprocessed=$WS/nnunet/nnUNet_preprocessed
export nnUNet_results=$WS/nnunet/nnUNet_results
E7=$WS/E7
S=$E7/scripts
TR=nnUNetTrainer_250epochs_noMirror
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results" "$E7/preds"

echo "=== 0. preflight ==="
find $WS -name '._*' -delete 2>/dev/null
test -d $nnUNet_raw/Dataset003_ParotidDirty || { echo "MISSING Dataset003"; exit 1; }
ls $E7/nnraw/imagesTs/*.nii.gz >/dev/null 2>&1 || { echo "MISSING E7 nnraw"; exit 1; }
ls $E7/e6npz/*.npz          >/dev/null 2>&1 || { echo "MISSING E7 e6npz"; exit 1; }
ls $E7/Dataset002_test/imagesTs/*.nii.gz >/dev/null 2>&1 || { echo "MISSING Dataset002 test"; exit 1; }
test -d $E7/models/nnunet_clean208/fold_0 || { echo "MISSING clean208 model"; exit 1; }
# register the custom no-mirror trainer (copy into nnunetv2's variants dir)
cp $S/nnUNetTrainer_250epochs_noMirror.py \
   "$(python3 -c 'import os,nnunetv2;print(os.path.join(os.path.dirname(nnunetv2.__file__),"training","nnUNetTrainer","variants"))')/" 2>/dev/null || true
python3 -c "from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_250epochs_noMirror import *; print('trainer OK')" \
  || { echo "trainer registration FAILED"; exit 1; }

echo "=== 1. build experts (Dataset005 left / Dataset006 right) ==="
python3 $S/build_experts_pod.py --raw $nnUNet_raw || exit 1

echo "=== 2. plan + preprocess ==="
nnUNetv2_plan_and_preprocess -d 5 -c 3d_fullres --verify_dataset_integrity || exit 1
nnUNetv2_plan_and_preprocess -d 6 -c 3d_fullres --verify_dataset_integrity || exit 1

echo "=== 3. train both experts in parallel (fold 0, noMirror, --npz) ==="
if [ ! -f $nnUNet_results/Dataset005_LeftExpert/${TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth ]; then
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 5 3d_fullres 0 -tr $TR --npz > $E7/train_left.log 2>&1 &
  PL=$!
else echo "left expert already trained, skip"; PL=0; fi
if [ ! -f $nnUNet_results/Dataset006_RightExpert/${TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth ]; then
  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 6 3d_fullres 0 -tr $TR --npz > $E7/train_right.log 2>&1 &
  PR=$!
else echo "right expert already trained, skip"; PR=0; fi
[ "$PL" != 0 ] && wait $PL; [ "$PR" != 0 ] && wait $PR
echo "training done"

echo "=== 4. predict experts (both-43 + single-side-58), disable_tta ==="
D2=$E7/Dataset002_test/imagesTs
SS=$E7/nnraw/imagesTs
nnUNetv2_predict -i $D2 -o $E7/preds/left_both  -d 5 -c 3d_fullres -tr $TR -f 0 --disable_tta
nnUNetv2_predict -i $SS -o $E7/preds/left_ss    -d 5 -c 3d_fullres -tr $TR -f 0 --disable_tta
nnUNetv2_predict -i $D2 -o $E7/preds/right_both -d 6 -c 3d_fullres -tr $TR -f 0 --disable_tta
nnUNetv2_predict -i $SS -o $E7/preds/right_ss   -d 6 -c 3d_fullres -tr $TR -f 0 --disable_tta

echo "=== 5. combine experts (hemifield + largest-CC) ==="
python3 $S/combine_experts.py --right-dir $E7/preds/right_both --left-dir $E7/preds/left_both --out $E7/preds/experts_both --cc
python3 $S/combine_experts.py --right-dir $E7/preds/right_ss   --left-dir $E7/preds/left_ss   --out $E7/preds/experts_ss   --cc

echo "=== 6. Part-1 predicts on the 58 single-side (the 4 existing models) ==="
python3 $S/predict_nnunet.py --in-dir $SS --out-dir $E7/preds/nnunet_clean208 \
        --model $E7/models/nnunet_clean208 --device cuda
for arm in e4 dirty430 masked430; do
  python3 $S/predict_e6.py --ckpt $E7/models/$arm.pth --preproc $E7/e6npz \
          --labels-ts $E7/nnraw/labelsTs --out $E7/preds/$arm --device cuda
done

echo "=== 7. FULL eval on pod (nothing heavy left for the Mac) ==="
RES=$E7/results; mkdir -p $RES
echo "--- experts on the both-parotid 43 (vs nnU-Net 0.8187) ---"
python3 $S/eval_testset.py --pred-dir $E7/preds/experts_both \
        --gt-dir $E7/Dataset002_test/labelsTs --model-name E7_experts_BOTH43 \
        --results-csv $RES/experts_both43.csv
echo "--- all models on the 58 single-side (annotated side + contralateral rate) ---"
python3 $S/eval_singleside.py --pred-root $E7/preds --gt-dir $E7/nnraw/labelsTs \
        --case-map $S/case_map.json --out-dir $RES | tee $RES/singleside_summary.txt
echo "--- contralateral montage ---"
python3 $S/make_montage.py --pred-root $E7/preds --img-dir $E7/nnraw/imagesTs \
        --gt-dir $E7/nnraw/labelsTs --case-map $S/case_map.json \
        --out $RES/contralateral_montage.png || echo "montage skipped (matplotlib?)"
echo "ALL E7 POD WORK DONE — download \$E7/preds, \$E7/results, and the two expert fold_0 dirs."
