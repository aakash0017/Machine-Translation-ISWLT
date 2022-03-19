#!/bin/bash -l
#SBATCH --job-name=numpy_compare
#SBATCH --output=example3.out
#
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH -p k2-hipri

# Without Numpy
echo "CALL #1:"
python ex3.py

# With Python2.7/Numpy1.11 and Python 2.7
echo "CALL #2:"
module load libs/numpy/1.11.3/gcc-4.8.5+atlas-3.10.3+setuptools-24.0.1+python-2.7.8
python ex3.py
module unload libs/numpy/1.11.3/gcc-4.8.5+atlas-3.10.3+setuptools-24.0.1+python-2.7.8

# With Python3.4/Numpy1.11 and Python 3.4
echo "CALL #3:"
module load apps/python3/3.4.3/gcc-4.8.5
module load libs/numpy_python34/1.11.3/gcc-4.8.5+atlas-3.10.3+python3-3.4.3
python3 ex3.py
module unload apps/python3/3.4.3/gcc-4.8.5
module unload libs/numpy_python34/1.11.3/gcc-4.8.5+atlas-3.10.3+python3-3.4.3

# With Python3.4/Numpy1.11 and Python 3.7
echo "CALL #4:"
module load apps/python3/3.7.4/gcc-4.8.5
python3 ex3.py