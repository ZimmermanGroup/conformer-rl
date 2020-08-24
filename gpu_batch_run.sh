#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=ppo_rtgn_pruning_lignin_curr_long
#ppo_rtgn_pruning_fix_lignin_curr
#SBATCH --mail-user=runxuanj@umich.edu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env
# The application(s) to execute along with its input arguments and options:
cd ~/conformer_generation/conformer-ml
module load cuda/10.1.243
module load gcc
python -u run_batch_train.py
