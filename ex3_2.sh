#!/bin/bash -l
#SBATCH --job-name=numpy_compare
#SBATCH --output=example3_2.out
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH -p k2-hipri

# Without Numpy
echo "CALL #1:"
python ex3.py

# With Python2.7/Numpy1.11 and Python 2.7
echo "CALL #2:"
module load apps/python3/3.7.4/gcc-4.8.5
source ~/Py37Env/numpy17/bin/activate
python3 ex3.py
deactivate

# With Python3.4/Numpy1.11 and Python 3.4
echo "CALL #3:"
source ~/Py37Env/numpy19/bin/activate
python3 ex3.py
deactivate