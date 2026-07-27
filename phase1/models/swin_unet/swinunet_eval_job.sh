#!/bin/bash
#PBS -N swinunet_eval
#PBS -l select=1:ncpus=4:ngpus=1:mem=40gb
#PBS -l walltime=01:00:00
#PBS -q gpu
#PBS -o /home/<hpc-user>/project/logs/swinunet_eval.out
#PBS -e /home/<hpc-user>/project/logs/swinunet_eval.err

exec > /home/<hpc-user>/project/logs/swinunet_eval.log 2>&1

source /apps/anaconda3/bin/activate deeplearning
cd /home/<hpc-user>/project/code

BEST_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk -F', ' '{print $2}')
export CUDA_VISIBLE_DEVICES=$BEST_GPU

echo "Starting Swin-UNet Evaluation on Physical GPU: $BEST_GPU"
python3 -u swinunet_evaluate.py
