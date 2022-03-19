#!/bin/bash -l
#SBATCH --job-name=example2
#SBATCH --output=example2.out
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH -p k2-hipri

####module load apps/python/2.7.8/gcc-4.8.5
python ex2.py

####module unload apps/python/2.7.8/gcc-4.8.5
module load apps/python3/3.7.4/gcc-4.8.5
python3 ex2.py