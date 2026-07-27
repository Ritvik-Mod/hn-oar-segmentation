#!/bin/bash
#PBS -N transunet
#PBS -l select=1:ncpus=4:ngpus=1:mem=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m abe
#PBS -o /home/<hpc-user>/project/logs/transunet_train.out
#PBS -e /home/<hpc-user>/project/logs/transunet_train.err

exec > /home/<hpc-user>/project/logs/transunet_live.log 2>&1

mkdir -p /home/<hpc-user>/project/checkpoints/transunet

source /apps/anaconda3/bin/activate deeplearning
cd /home/<hpc-user>/project/code

echo "Job started: $(date)"
echo "Running on: $(hostname)"

# OOM memory fragmentation safeguard
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

python3 -u transunet_train.py

echo "Job finished: $(date)"
