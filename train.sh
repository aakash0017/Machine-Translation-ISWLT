#!/bin/bash -l
#SBATCH -N 1 #1 node
#SBATCH -n 8
#SBATCH --job-name="isometric_iswlt"
#SBATCH --output=isometric_iswlt.out
#SBATCH --error=isometric_iswlt.err
#
#SBATCH --ntasks=8
#SBACTH
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00 

python3 train.py