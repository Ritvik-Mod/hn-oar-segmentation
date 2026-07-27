#!/usr/bin/env bash
# E6 on the pod — 2× GPU, both arms in parallel, then predict+eval (raw & +CC).
# Upload this dir + preprocessed/ to /workspace/E6, then:  bash pod_run_e6.sh
# Expects: /workspace/E6/{*.py, preprocessed/{train,test}}, and pipeline/ + Dataset002 labelsTs
# reachable at the ../../ relative paths the scripts use (see UPLOAD below).
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

echo "=== deps ==="
pip install -q nibabel scipy scikit-learn 2>&1 | tail -2
find /workspace -name '._*' -delete 2>/dev/null || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'nGPU', torch.cuda.device_count())"

GT="$HERE/labelsTs"        # Dataset002 labelsTs, uploaded alongside (see UPLOAD)
EVAL="$HERE/eval_testset.py"

echo "=== TRAIN both arms in parallel (GPU0=masked, GPU1=dirty) ==="
CUDA_VISIBLE_DEVICES=0 python3 train_e6.py --loss masked \
    --preproc ./preprocessed/train --save-dir ./ckpt_masked430 > train_masked430.log 2>&1 &
PID_M=$!
CUDA_VISIBLE_DEVICES=1 python3 train_e6.py --loss dirty \
    --preproc ./preprocessed/train --save-dir ./ckpt_dirty430  > train_dirty430.log 2>&1 &
PID_D=$!
echo "masked PID=$PID_M (GPU0)  dirty PID=$PID_D (GPU1) — tail -f train_*.log to watch"
wait $PID_M; echo "masked done (exit $?)"
wait $PID_D; echo "dirty done (exit $?)"

run_eval () {  # $1=arm-name $2=ckpt-dir
  local arm="$1" ck="$2"
  echo "=== predict+eval $arm ==="
  CUDA_VISIBLE_DEVICES=0 python3 predict_e6.py --ckpt "$ck/best_model.pth" \
      --preproc ./preprocessed/test --out "./pred_$arm" --labels-ts "$GT"
  python3 "$EVAL" --pred-dir "./pred_$arm" --gt-dir "$GT" \
      --model-name "Custom_3D_UNet_$arm" --results-csv ./eval.csv
  python3 postproc_largestcc.py --pred-dir "./pred_$arm" --out "./pred_${arm}_cc"
  python3 "$EVAL" --pred-dir "./pred_${arm}_cc" --gt-dir "$GT" \
      --model-name "Custom_3D_UNet_${arm}_CC" --results-csv ./eval.csv
}

run_eval masked430 ./ckpt_masked430
run_eval dirty430  ./ckpt_dirty430

echo "=== DONE — eval.csv ==="
cat ./eval.csv
