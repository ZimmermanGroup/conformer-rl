# conformer-rl
An open-source deep reinforcement learning library for conformer generation.

## Documentation
Documentation can be found at <conformer-rl.readthedocs.io>.

## Installation

* Prerequisites
  * Install RDKit

        $ conda install -c conda-forge rdkit

  * Install PyTorch Geometric. Since the installation is heavily dependent on the PyTorch, OS and CUDA versionsof the system, detailed instructions for installing PyTorch Geometric can be found at <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>.

* Install conformer-rl

        $ pip install conformer-rl

* Verify Installation
As a quick check to verify the installation has succeeded, navigate to the `examples` directory
and run `test_example.py`. The script should finish running in a few minutes or less. If no errors ware encountered
then most likely the installation has succeeded.

## Features

* Agents - `conformer_rl` contains implementations of agents for several deep reinforcement learning algorithms,
including recurrent and non-recurrent versions of A2C and PPO. `conformer_rl` also includes a base agent
interface `base_agent` for constructing new agents.

* Models - Implementations of various graph neural network models are included. Each model is compatible with
any molecule.

* Analysis - `conformer_rl` contains a module for visualizing metrics and molecule conformers in Jupyter/IPython notebooks.
`examples/analysis.ipynb` shows some examples on how the visualizing tools can be used.

## Quick Start
The `examples` directory contain several scripts for training on pre-built agents and environments.
Visit the [Quick Start]() to get started.
