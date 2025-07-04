#!/bin/bash
#SBATCH -p serc,gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --time=48:00:00
#SBATCH --output=abc-%j.out

ml py-pytorch/2.0.0_py39
ml load viz py-matplotlib/3.7.1_py39

python3 semantic.py
