#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=lignin_three
#SBATCH --mail-user=tgog@umich.edu
#SBATCH --cpus-per-task=35
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env
# The application(s) to execute along with its input arguments and options:
source deactivate my-rdkit-env
source ~/.bashrc
cd ~/conformer-ml/
module load cuda
module load gcc
python run_batch_train.py
