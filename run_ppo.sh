#!/bin/bash

# The interpreter used to execute the script


#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=eval
#SBATCH --mail-user=runxuanj@umich.edu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2048m
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env
# The application(s) to execute along with its input arguments and options:
conda activate my-rdkit-env
cd ~/conformer-ml-testing/conformer-ml
module load cuda
module load gcc
python drl-oneset.py
