#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=eval_lignin_cpu_timing
#SBATCH --mail-user=tgog@umich.edu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096m
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --get-user-env
# The application(s) to execute along with its input arguments and options:
source deactivate my-rdkit-env
source ~/.bashrc
cd ~/conformer-ml/
module load gcc
# python transformer_test.py
# python run_eval.py
# python obabel.py
python run_eval_cpu.py
