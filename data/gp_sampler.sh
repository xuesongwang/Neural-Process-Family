#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=8gb
#SBATCH --ntasks-per-node=4
#SBATCH --account=OD-228587

module load pytorch/1.8.1-py39-cuda112-mpi
#python -m venv --system-site-packages/scratch1/wan410/venv                           # create the virtual environment
source /scratch1/wan410/venv/bin/activate                                             # use the virtual environment
#python -m pip install --upgrade pip wheel                                             # system pip is too elderly
#python -m pip install -r requirements.txt                                             # actual stuff this code needs

python3 GP_sampler_NPF.py

