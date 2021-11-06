# conformer-rl
An open-source deep reinforcement learning library for conformer generation.

[![Documentation Status](https://readthedocs.org/projects/conformer-rl/badge/?version=latest)](https://conformer-rl.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/conformer-rl.svg)](https://badge.fury.io/py/conformer-rl)

## Documentation
Documentation can be found at https://conformer-rl.readthedocs.io/.

## Platform Support
Since conformer-rl can be run within a Conda environment, it should work on all platforms (Windows, MacOS, Linux).

## Installation
* We recommend installing in a new Conda environment.
  * If you are new to using Conda, you can install it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and learn more about environments [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

* Install dependencies
  * Install [PyTorch](https://pytorch.org/get-started/locally/). PyTorch version of 1.8.0 or greater is required for conformer-rl.
  * Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
    * **Important Note**:  Please make sure to use the same package installer for installing both PyTorch and PyTorch geometric. 
  
      For example, if you installed PyTorch with pip, use the [pip instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels) for installing PyTorch Geometric. Similarly, if you installed PyTorch with Conda, use the [Conda instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-anaconda) for installing PyTorch Geometric. Otherwise you may run into errors such as "undefined symbol" when using PyTorch Geometric.
  * Install RDKit by running the following command
    ```
    conda install -c conda-forge rdkit
    ```

* Install conformer-rl
  ```
  pip install conformer-rl
  ```

  * This will automatically install the additional dependencies needed for conformer-rl. Note that conformer-rl requires Python >= 3.7.

### Verify Installation:

  As a quick check to verify the installation has succeeded, navigate to the [examples](https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples) directory
  and run `base_example.py`. The script should finish running in a few minutes or less. If no errors ware encountered then most likely the installation has succeeded.

## Additional Installation for Analysis/Visualization Tools

  Some additional dependencies are required for visualizing molecules in Jupyter/IPython notebooks. 
  
  Firstly, install `jupyterlab`, `py3Dmol`, and `seaborn` (these should already be installed after installing conformer-rl):
  ```
  pip install jupyterlab py3Dmol seaborn
  ```
  Install `nodejs`. This is only required for creating interactive molecule visualizations in Jupyter:
  ```
  conda install nodejs
  ```
  Install the [jupyterlab_3dmol](https://github.com/3dmol/jupyterlab_3Dmol) extension for visualizing molecules interactively in Jupyter:
  ```
  jupyter labextension install jupyterlab_3dmol
  ```
  You should now be able to use the analysis components of conformer-rl for generating figures and visualizing molecule in Jupyter. To test that the installation was succesful, try running the example Jupyter notebook:
  ```
  jupyter-lab examples/example_analysis.ipynb
  ```

## Features

* Agents - `conformer_rl` contains implementations of agents for several deep reinforcement learning algorithms,
including recurrent and non-recurrent versions of A2C and PPO. `conformer_rl` also includes a base agent
interface BaseAgent for constructing new agents.

* Models - Implementations of various graph neural network models are included. Each model is compatible with
any molecule.

* Environments - Implementations for several pre-built environments that are compatible with any molecule. Environments are built
on top of the modularized ConformerEnv interface, making it easy to create custom environments
and max-and-match different environment components.

* Analysis - `conformer_rl` contains a module for visualizing metrics and molecule conformers in Jupyter/IPython notebooks.
The [example notebook](https://drive.google.com/drive/folders/1WAnTv4SGwEQHHqyMcbrExzUob_mOfTcM?usp=sharing) in the [examples](https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples) directory shows some examples on how the visualizing tools can be used.

## Quick Start
The [examples](https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples) directory contain several scripts for training on pre-built agents and environments.
Visit [Quick Start](https://conformer-rl.readthedocs.io/en/latest/tutorial/quick_start.html) to get started.

## Issues and Feature Requests
We are actively adding new features to this project and are open to all suggestions. If you believe you have encountered a bug, or if you have a feature that you would like to see implemented, please feel free to file an [issue](https://github.com/ZimmermanGroup/conformer-rl/issues).

## Developer Documentation
Pull requests are always welcome for suggestions to improve the code or to add additional features. We encourage new developers to document new features and write unit tests (if applicable). For more information on writing documentation and unit tests, see the [developer documentation](https://conformer-rl.readthedocs.io/en/latest/developer.html).
