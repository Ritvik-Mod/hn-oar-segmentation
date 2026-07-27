#!/bin/bash
#PBS -N att_eval
#PBS -l select=1:ncpus=8:mem=40gb
#PBS -l walltime=4:00:00
#PBS -q workq
#PBS -o /home/<hpc-user>/project/logs/attention_eval.out
#PBS -e /home/<hpc-user>/project/logs/attention_eval.err

exec > /home/<hpc-user>/project/logs/attention_eval.log 2>&1

source /apps/anaconda3/bin/activate deeplearning
cd /home/<hpc-user>/project/code

python3 -u attention_evaluate.py
