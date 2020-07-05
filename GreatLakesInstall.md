## Instructions for installing and using TorsionNet on GreatLakes cluster

**Part 1: prerequisites**
1. SSH into the Great Lakes cluster:
    - `ssh <username>@greatlakes.arc-ts.umich.edu`
2. Load dependencies:
    - `module load gcc opencv cuda`
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
        `conda env activate my-rdkit-env`

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
2. Run `python drl-ppo.py` for around a minute and see that no import errors occur.
3. Run a job on Great Lakes:
    - `sbatch run_ppo.sh`
    - verify that no import errors occur after running the job for a few minutes
    - make sure to delete the job after verifying on greatlakes.arc-ts.umich.edu
