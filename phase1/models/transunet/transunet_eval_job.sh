#!/bin/bash
#PBS -N transunet_eval
#PBS -l select=1:ncpus=4:ngpus=1:mem=40gb
#PBS -l walltime=01:00:00
#PBS -q gpu
#PBS -o /home/<hpc-user>/project/logs/transunet_eval.out
#PBS -e /home/<hpc-user>/project/logs/transunet_eval.err

exec > /home/<hpc-user>/project/logs/transunet_eval.log 2>&1

source /apps/anaconda3/bin/activate deeplearning
cd /home/<hpc-user>/project/code

# ── THE HPC HACK: Find the emptiest GPU and force PyTorch to use it ──
# nvidia-smi outputs: "free_memory, index" -> sorts by highest memory -> grabs the index
BEST_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
export CUDA_VISIBLE_DEVICES=$BEST_GPU

echo "=========================================================="
echo "🛡️ PyTorch is securely locked to Physical GPU: $BEST_GPU"
echo "=========================================================="
nvidia-smi
echo "=========================================================="

echo "Starting Evaluation..."
python3 -u transunet_evaluate.py
