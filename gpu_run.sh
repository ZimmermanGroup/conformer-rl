#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=eval_lignin_long_save
#SBATCH --mail-user=tgog@umich.edu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096m
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=05:00:00
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
# python transformer_test.py
# python run_eval.py
# python obabel.py
python t_chains_eval.py
