#!/bin/bash
#SBATCH --job-name=video
#SBATCH --partition=general
#SBATCH --time=10:00:00  # Max job run time
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=50GB
#SBATCH --nodes=1


conda activate asl-ez
CUDA_VISIBLE_DEVICES=0 python train.py