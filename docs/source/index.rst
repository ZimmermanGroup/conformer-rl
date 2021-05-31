**************************
conformer_rl Documentation
**************************

.. conformer_rl documentation master file, created by
   sphinx-quickstart on Wed Apr  7 14:36:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tutorial
========
.. toctree::
   :maxdepth: 1

   tutorial/quick_start
   tutorial/customizing_env_basic
   tutorial/customizing_env_advanced

API Reference
=============
.. toctree::
   :maxdepth: 2

   agents/agents
   models/models
   config/config
   environments/environments
   molecule_generation/molecule_generation
   logging/logging
   analysis/analysis
   utils/utils

Introduction
============
:mod:`conformer_rl` is an open-source deep reinforcement learning library for conformer generation, written in Python using 
`PyTorch <https://pytorch.org/>`_ and `RDKit <https://www.rdkit.org/>`_.

Source code can be found on GitHub: https://github.com/ZimmermanGroup/conformer-rl.

Installation
------------

Prerequisites
^^^^^^^^^^^^^
* Install :mod:`rdkit`::

   $ conda install -c conda-forge rdkit

* Install PyTorch Geometric. Since the installation is heavily dependent on the PyTorch, OS and CUDA versions
  of the system, detailed instructions for installing PyTorch Geometric can be found at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

Finally, install :mod:`conformer_rl`::

   $ pip install conformer-rl

Verify Installation
^^^^^^^^^^^^^^^^^^^
As a quick check to verify the installation has succeeded, navigate to the ``examples`` directory
and run ``test_example.py``. The script should finish running in a few minutes or less. If no errors ware encountered
then most likely the installation has succeeded.

Features
========

`Agents`_
---------
:mod:`conformer_rl` contains implementations of agents for several deep reinforcement learning algorithms,
including recurrent and non-recurrent versions of A2C and PPO. :mod:`conformer_rl` also includes a base agent
interface :mod:`conformer_rl.agents.base_agent` for constructing new agents.

`Models`_
---------
Implementations of various graph neural network models are included. Each model is compatible with
any molecule.

`Environments`_
---------------
:mod:`conformer_rl` contains several pre-built environments that are compatible with any molecule. Environments are built
on top of the modularized :mod:`conformer_rl.environments.conformer_env` interface, making it easy to create custom environments
and max-and-match different environment components.

`Analysis`_
-----------
:mod:`conformer_rl` contains a module for visualizing metrics and molecule conformers in Jupyter/IPython notebooks.
``examples/analysis.ipynb`` shows some examples on how the visualizing tools can be used.


Examples
========
The ``examples`` directory contain several scripts for training on pre-built agents and environments.
Visit the `Tutorial`_ to get started.

