#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 0-02:59:00
#SBATCH -p core
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=2G



       --global_variables "output_dir:<path>/icolos/tests/junk" -debug

