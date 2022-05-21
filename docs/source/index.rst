************
Introduction
************

.. conformer_rl documentation master file, created by
   sphinx-quickstart on Wed Apr  7 14:36:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :caption: Tutorial
   :hidden:
   :maxdepth: 2

   tutorial/install
   tutorial/customizing_env_basic
   tutorial/customizing_env_advanced

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   agents/agents
   models/models
   config/config
   environments/environments
   molecule_generation/molecule_generation
   logging/logging
   analysis/analysis
   utils/utils

.. toctree::
   :caption: Developer Documentation
   :hidden:
   :maxdepth: 2

   developer


:mod:`conformer_rl` is an open-source deep reinforcement learning library for conformer generation, written in Python using 
`PyTorch <https://pytorch.org/>`_ and `RDKit <https://www.rdkit.org/>`_.

Source code can be found on GitHub: https://github.com/ZimmermanGroup/conformer-rl.


Features
--------

:ref:`Agents`
^^^^^^^^^^^^^
:mod:`conformer_rl` contains implementations of agents for several deep reinforcement learning algorithms,
including recurrent and non-recurrent versions of A2C and PPO. :mod:`conformer_rl` also includes a base agent
interface :mod:`~conformer_rl.agents.base_agent` for constructing new agents.

:ref:`Models`
^^^^^^^^^^^^^
Implementations of various graph neural network models are included. Each model is compatible with
any molecule.

:ref:`Environments`
^^^^^^^^^^^^^^^^^^^
:mod:`conformer_rl` contains several pre-built environments that are compatible with any molecule. Environments are built
on top of the modularized :mod:`~conformer_rl.environments.conformer_env` interface, making it easy to create custom environments
and mix-and-match different environment components.

:ref:`Analysis`
^^^^^^^^^^^^^^^
:mod:`conformer_rl` contains a module for visualizing metrics and molecule conformers in Jupyter/IPython notebooks.
The `example notebook <https://drive.google.com/drive/folders/1WAnTv4SGwEQHHqyMcbrExzUob_mOfTcM?usp=sharing>`_ shows some examples on how the visualizing tools can be used.


Examples
--------
The `examples <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples>`_ directory contains several scripts for training on pre-built agents and environments.
Visit the :ref:`Quick Start` to get started.
