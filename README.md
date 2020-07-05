# TorsionNet
## Installation 

The libraries needed are included as a conda environment file, and can be installed via 
>`conda env create -f environment.yml`

This repo must be used in conjunction with the agent library [DeepRL](https://github.com/tarungog/DeepRL). Install via cloning and run command inside directory: 
>`pip install -e .`

To play with lignin generation, the [lignin-kmc](https://github.com/michaelorella/lignin-kmc) library must also be installed.

## Runtime

The code is meant to be run on the Great Lakes cluster, although can be easily modified to run on other compute grids. The key Slurm script to run a training job is located at `gpu_batch_run.sh`, and all other scripts are based off of this one.

It calls the python file `run_batch_train.py`, which is where all details of the experiment must be set before running the job script. Here, we set the train and validation gym environments, along with the algorithmic hyperparameters.

All gym environments are written in `graphenvironments.py`. To make a new environment, subclass `PruningSetGibbs` and modify as needed. There are many examples in the file to learn from.

Every model used in these experiments is written in PyTorch, and all model architectures can be found in `models.py`.

## Other notable files

- `hydrocarbons.py` is the molecular generation file for branched alkanes
- `zeke_oligomers.ipynb` is the notebook used for lignin mol file generation
