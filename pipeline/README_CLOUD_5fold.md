# 5-fold ensemble on a 4-GPU RunPod box

Trains folds 1–4 in parallel (one per GPU), then ensembles all 5 (fold 0 already
trained) on the held-out test set. Custom trainer `nnUNetTrainer_250epochs_noMirror`
(mirroring OFF — never change this).

Placeholders: `IP` / `PORT` from RunPod Connect → "SSH over exposed TCP".

## A. Rent the box
- RunPod → Deploy Pod. **Compute → GPU count = 4.** GPU type: **RTX 4090** (or A5000/A40).
- Template: RunPod PyTorch. SSH key added. Disk/volume ≥ 60 GB.
- **If your previous network volume still exists (Storage tab), attach it** — then
  `Dataset002` + its preprocessing + fold_0 are already there and you skip the 7 GB upload.
- Deploy → copy the SSH-over-TCP IP/PORT.
- Cost check: 4× 4090 ≈ $1.4–2.8/hr; ~2 h ⇒ ~$3–6.

## B. Connect + check what's on the volume
```bash
ssh root@IP -p PORT -i ~/.ssh/id_ed25519
```
```bash
export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results
ls $nnUNet_raw/Dataset002_Parotid            # dataset present?
ls $nnUNet_preprocessed/Dataset002_Parotid   # preprocessing cached?
find $nnUNet_results/Dataset002_Parotid -name "checkpoint_final.pth"  # fold_0?
nvidia-smi --query-gpu=index,name --format=csv,noheader   # confirm 4 GPUs
```

## C. Setup (always) + upload (only if the volume was empty)
Always:
```bash
pip install -q nnunetv2 nibabel scipy scikit-image
mkdir -p /workspace/pipeline
NNUNET_DIR=$(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
```
Register the custom trainer (paste the file if not present):
```bash
cat > $NNUNET_DIR/training/nnUNetTrainer/variants/nnUNetTrainer_250epochs_noMirror.py <<'EOF'
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)
class nnUNetTrainer_250epochs_noMirror(nnUNetTrainerNoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
EOF
python3 -c "from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_250epochs_noMirror import nnUNetTrainer_250epochs_noMirror; print('trainer OK')"
```
ONLY if the volume was empty — upload from your Mac (new tab):
```bash
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
  ~/Desktop/Dataset002_Parotid/ root@IP:/workspace/nnunet/nnUNet_raw/Dataset002_Parotid/
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
  ~/Desktop/nnunet_results_Dataset002_noMirror/Dataset002_Parotid/ \
  root@IP:/workspace/nnunet/nnUNet_results/Dataset002_Parotid/
cd "/Users/ritvikmod/Desktop/Projects/ct scan project"
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" pipeline/ root@IP:/workspace/pipeline/
```
Then on the pod, preprocess (only if `nnUNet_preprocessed/Dataset002_Parotid` was missing):
```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
```

## D. Train folds 1–4 IN PARALLEL (one GPU each)
```bash
cat > /workspace/launch_fold.sh <<'EOF'
#!/bin/bash
export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results
export CUDA_VISIBLE_DEVICES=$2
nnUNetv2_train 2 3d_fullres $1 -tr nnUNetTrainer_250epochs_noMirror --npz 2>&1 | tee /workspace/train_fold$1.log
EOF
chmod +x /workspace/launch_fold.sh
tmux new-session -d -s fold1 "/workspace/launch_fold.sh 1 0"
tmux new-session -d -s fold2 "/workspace/launch_fold.sh 2 1"
tmux new-session -d -s fold3 "/workspace/launch_fold.sh 3 2"
tmux new-session -d -s fold4 "/workspace/launch_fold.sh 4 3"
tmux ls          # should list fold1..fold4
```
Monitor:
```bash
for f in 1 2 3 4; do echo "== fold $f =="; tail -n 2 /workspace/train_fold$f.log; done
nvidia-smi       # all 4 GPUs busy
```
Each finishes in ~1–2 h and ends with `Mean Validation Dice`.

## E. Ensemble + evaluate (after all 4 logs finish)
```bash
export nnUNet_raw=/workspace/nnunet/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed
export nnUNet_results=/workspace/nnunet/nnUNet_results

nnUNetv2_predict -i $nnUNet_raw/Dataset002_Parotid/imagesTs \
  -o /workspace/pred_ensemble -d 2 -c 3d_fullres \
  -tr nnUNetTrainer_250epochs_noMirror -f 0 1 2 3 4 --disable_tta

cd /workspace/pipeline
python eval_testset.py --pred-dir /workspace/pred_ensemble \
  --gt-dir $nnUNet_raw/Dataset002_Parotid/labelsTs \
  --model-name nnUNet_5fold_ensemble --results-csv /workspace/results_5fold_TEST.csv
```
(Optional, needs all 5 folds' validation softmax — only if fold_0 was retrained here:
`nnUNetv2_find_best_configuration 2 -c 3d_fullres -tr nnUNetTrainer_250epochs_noMirror`
then apply the postprocessing command it prints before eval.)

## F. Download then terminate
```bash
# on Mac
rsync -avP -e "ssh -p PORT -i ~/.ssh/id_ed25519" --exclude 'validation/' \
  root@IP:/workspace/nnunet/nnUNet_results/Dataset002_Parotid \
  ~/Desktop/nnunet_results_Dataset002_5fold
scp -P PORT -i ~/.ssh/id_ed25519 -r root@IP:/workspace/pred_ensemble ~/Desktop/pred_ensemble
```
Then Terminate the pod.
