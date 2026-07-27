#!/bin/bash
#PBS -N attention_parotid
#PBS -l select=1:ncpus=4:ngpus=1:mem=40gb
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -m abe
#PBS -o /home/<hpc-user>/project/logs/attention_train.out
#PBS -e /home/<hpc-user>/project/logs/attention_train.err

exec > /home/<hpc-user>/project/logs/attention_live.log 2>&1

echo "Job started: $(date)"
echo "Starting job on compute node: $(hostname)"

source /apps/anaconda3/bin/activate deeplearning
cd /home/<hpc-user>/project/code
nvidia-smi

echo "Firing up PyTorch..."
python3 -u attention_train.py

echo "Job finished: $(date)"
