#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=crunching_rdkit
#SBATCH --mail-user=tgog@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=36
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
python hydrocarbons.py
