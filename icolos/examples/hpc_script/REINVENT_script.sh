#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=28G
#SBATCH --time=346:0:0
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

