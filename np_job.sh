#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem=8gb
#SBATCH --ntasks-per-node=4
#SBATCH --account=OD-228587
#SBATCH --gres=gpu:1

module load pytorch/2.0.0-py39-cuda121-mpi
#python -m venv --system-site-packages/scratch1/wan410/venv                           # create the virtual environment
source /scratch1/wan410/venv/bin/activate                                             # use the virtual environment
#python -m pip install --upgrade pip wheel                                             # system pip is too elderly
#python -m pip install -r requirements.txt                                             # actual stuff this code needs


#python3 NP_or_ANP_train_newGP_task.py --kernel1 0 --kernel2 1
python3 NP_or_ANP_train_newGP_task.py --kernel1 0 --kernel2 1 --modelname NP
#python3 NP_or_ANP_train_newGP_task.py --kernel1 0 --kernel2 2

