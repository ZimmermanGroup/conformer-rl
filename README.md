# conformer-rl
An open-source deep reinforcement learning library for conformer generation.

[![Documentation Status](https://readthedocs.org/projects/conformer-rl/badge/?version=latest)](https://conformer-rl.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/conformer-rl.svg)](https://badge.fury.io/py/conformer-rl)

## Documentation
Documentation can be found at https://conformer-rl.readthedocs.io/.

## Installation
* We recommend installing in a new conda environment.
  * If you are new to using conda, you can install it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and learn more about environments [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
  * Create and activate a new environment:
    ```
    $ conda create --name conformerrl # create a new environment
    $ conda activate conformerrl # activate the new environment
    ```
* Install dependencies
  * Install RDKit

        $ conda install -c conda-forge rdkit

  * We recommend installing the dependencies and versions listed in `requirements.txt`:
    ```
    $ pip install -r requirements.txt
    ```
    The library will most likely still work if you use a different version than what is listed in `requirements.txt`, but most testing was done using these versions.

* Install conformer-rl

        $ pip install conformer-rl

  * If you did not install dependencies using `requirements.txt`, you will need to manually install Pytorch Geometric [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

* Verify Installation: <br />
As a quick check to verify the installation has succeeded, navigate to the [examples](https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples) directory
and run `base_example.py`. The script should finish running in a few minutes or less. If no errors ware encountered
then most likely the installation has succeeded.

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
The [example notebook](https://colab.research.google.com/drive/1Y6u4fFM4BkGLtxetZ0QWbR5sZO1U1KPr) in the [examples](https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples) directory shows some examples on how the visualizing tools can be used.

## Quick Start
The [examples](https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples) directory contain several scripts for training on pre-built agents and environments.
Visit [Quick Start](https://conformer-rl.readthedocs.io/en/latest/tutorial/quick_start.html) to get started.

