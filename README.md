# TorsionNet
Deep Reinforcement Learning for Conformer Generation

## Installation in Great Lakes Cluster
**Part 1: Prerequisites**
1. SSH into the Great Lakes cluster:
    - `ssh <username>@greatlakes.arc-ts.umich.edu`
2. Load dependencies:
    - `module load gcc opencv cuda/10.1.243`
3. Install conda on Great Lakes if not already installed:
    - `wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh`
    - `chmod u+x Anaconda3-2020.02-Linux-x86_64.sh`
    - `./Anaconda3-2020.02-Linux-x86_64.sh`
    - follow the prompts that appear in the terminal
4. Clone the conformer-ml repository onto Great Lakes:
    - link: https://github.com/ZimmermanGroup/conformer-ml
5. Navigate to the conformer-ml directory and create the new conda environment from the environment.yml file:
    - `conda env create -f environment.yml`
    - to activate the environment:
        > conda env activate my-rdkit-env
6. To play with lignin generation, the [lignin-kmc](https://github.com/michaelorella/lignin-kmc) library must also be installed.

**Part 2: Install missing dependencies**
1. Install DeepRL:
    - Make sure the my-rdkit-env environment is activated
    - Clone the DeepRL repository from this link: https://github.com/runxuanjiang/DeepRL
    - Navigate into the DeepRl repository and run the following command:
        > pip install -e .
2. Install PyTorch Geometric:
    - `pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-geometric`
    
**Part 3: Verify that environment is set up properly**
1. Check that the current directory is the conformer-ml directory and the my-rdkit-env environment is active.
2. Run `run_batch_train.py` for around a minute and see that no import errors occur.
3. Run a job on Great Lakes:
    - `sbatch gpu_batch_run.sh`
    - verify that no import errors occur after running the job for a few minutes
    - make sure to delete the job after verifying on greatlakes.arc-ts.umich.edu


## Runtime
The code is meant to be run on the Great Lakes cluster, although can be easily modified to run on other compute grids. The key Slurm script to run a training job is located at `gpu_batch_run.sh`, and all other scripts are based off of this one.

It calls the python file `run_batch_train.py`, which is where all details of the experiment must be set before running the job script. Here, we set the train and validation gym environments, along with the algorithmic hyperparameters.

**For creating a custom environment**
Create a new directory for the environment. The directory should contain a .json file for each molecule in the environment. The json file should contain the following parameters:
- mol (String)
    - required: Conditional - only if 'molfile' is not defined
    - contains the molecule in smiles format
- molfile (String)
    - required: Conditional - only if 'mol' is not defined
    - contains the name of the .mol file containing the molecule
- standard (double)
    - required: Yes
    - contains the standard energy for the molecule (minimum conformer energy of the molecule)
- total (double)
    - required: No (defaults to 1)
    - total = np.sum(np.exp(-(energys-standard)))
- inv_temp (double):
    - required: No (defaults to 1)

Example:
`{"standard": 7.668625034772399, "total": 13.263723987526067, "mol": "CC(CCC)CCCC(CCCC)CC"}`

## File Organization
**agents**
- Contains implementations dependent on DeepRL library for different reinforcement learning agents modified to accomodate molecule graph structures (GNN's) and recurrent neural networks (RNN's).
- *Implementations for PPO and A2C to be added.*

**environment**
- graphEnvironments.py contains the gym environment used for conformer generation.

**generateMolecule**
- contains files for generating molecule JSON files for an environment.
- *Additional implementations and more robust options to be added.*

**models**
- recurrentTorsionGraphNetBatch.py contains the neural network used for training.

**notebooks**
- contains miscellaneous notebooks for testing and training for TorsionNet paper.
- testing.ipynb contains experiments with rdkit and molecule editing/visualization.
- zeke_oligomers.ipynb contains experiments for lignin mol file generation.

**utils**
- agentUtilities.py contains tools for interacting with deepRl agent implementations.
- moleculeToVector.py contains functions for converting a rdkit mol representation to a graph data structure input for graph neural nets for PyTorch Geometric.
- moleculeUtilites.py contains functions for modifying and visualizing molecules.

**run_batch_train.py**
- main executable file that trains the model based on selected environments. Contains hyperparameters for agent.
