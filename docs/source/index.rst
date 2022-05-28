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
   tutorial/getting_started
   tutorial/model_tuning
   tutorial/customizing_env_1
   tutorial/customizing_env_2

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


:mod:`conformer_rl` is an open-source deep reinforcement learning library for conformer generation, written in Python using `PyTorch <https://pytorch.org/>`_ and `RDKit <https://www.rdkit.org/>`_.

Source code can be found on GitHub: https://github.com/ZimmermanGroup/conformer-rl.

What is Conformer Generation?
=============================
Most covalently bonded molecules (including proteins and most drug-like molecules) can take on a variety of different shapes, or conformations. The task of conformer generation involves generating a representative set of likely (low-energy) conformers that the molecule can take on. Efficient generation of such conformations is useful for a variety of applications in chemistry.

:mod:`conformer_rl` is an open-source software library for using deep reinforcement learning methods in conformer generation. 


Features
========

Scripts and Examples
--------------------
Using :mod:`conformer_rl`'s API, a reinforcement learning agent can be trained in generating conformers for a molecule given only that molecule as input. Several examples of training scripts can be found in the examples directory, which can be easily modified to set up a training script for your own molecule. For more details on setting up a Python training script see :ref:`Getting Started - Training a Conformer Generation Agent`.

Reinforcement Learning components
---------------------------------

:ref:`Agents`
^^^^^^^^^^^^^
:mod:`conformer_rl` contains implementations of several state-of-the-art algorithms for training reinforcement learning agents on a task, including recurrent and non-recurrent versions of A2C and PPO. :mod:`conformer_rl` also includes a base agent interface :mod:`~conformer_rl.agents.base_agent` for constructing new agents.

:ref:`Models`
^^^^^^^^^^^^^
Implementations of various graph neural network models are included.

:ref:`Environments`
^^^^^^^^^^^^^^^^^^^
:mod:`conformer_rl` contains several pre-built reinforcement learning environments that simulate the conformer generation task for any covalently bonded molecule. Many environments are similar to the conformer generation environment described in [1], where in each episode, conformers for the molecule are sequentially generated in each step. However, environments are built on top of the modularized :mod:`~conformer_rl.environments.conformer_env` interface, making it easy to create custom environments and mix-and-match different environment components.

:ref:`Analysis`
^^^^^^^^^^^^^^^
:mod:`conformer_rl` contains a module for visualizing metrics and molecule conformers in Jupyter/IPython notebooks. The `example notebook <https://drive.google.com/drive/folders/1WAnTv4SGwEQHHqyMcbrExzUob_mOfTcM?usp=sharing>`_ shows some examples on how the visualizing tools can be used.


Getting Started
===============
The `examples <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples>`_ directory contains several scripts for training on pre-built agents and environments. See :ref:`Installation` on how to install :mod:`conformer_rl`. See the :ref:`Getting Started - Training a Conformer Generation Agent` section to learn how to train an agent on your own custom molecule.

.. [1] `TorsionNet <https://arxiv.org/abs/2006.07078>`_