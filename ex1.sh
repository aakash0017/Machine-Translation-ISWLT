#!/bin/bash -l
#SBATCH --job-name=example1
#SBATCH --output=example1.out
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH -p k2-hipri

module load apps/python3/3.7.4/gcc-4.8.5

python ex1.py