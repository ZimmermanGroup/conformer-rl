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

   tutorial/quick_start
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

Installation
------------

Prerequisites
^^^^^^^^^^^^^
* We recommend installing in a new Conda environment.

   * If you are new to using Conda, you can install it `here <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ and learn more about environments `here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

* Install dependencies

   * Install `PyTorch`_ . PyTorch version of 1.8.0 or greater is required for :mod:`conformer_rl`.
   * Install `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ .

      * **Important Note**: Please make sure to use the same package installer (i.e., Conda, Pip) for installing both PyTorch and PyTorch geometric.

   * Install `RDKit <https://www.rdkit.org/>`_::

      conda install -c conda-forge rdkit

* Finally, install :mod:`conformer_rl`::

   pip install conformer-rl

Verify Installation
^^^^^^^^^^^^^^^^^^^
As a quick check to verify the installation has succeeded, navigate to the ``examples`` directory
and run `base_example.py <https://github.com/ZimmermanGroup/conformer-rl/blob/master/examples/base_example.py>`_. The script should finish running in a few minutes or less. If no errors are encountered
then most likely the installation has succeeded.

Additional Installation for Analysis/Visualization Tools
--------------------------------------------------------
Some additional dependencies are required for visualizing molecules in Jupyter/IPython notebooks.

Firstly, install :code:`jupyterlab`, :code:`py3Dmol`, and :code:`seaborn` (these should already be installed after installing conformer-rl)::

   pip install jupyterlab py3Dmol seaborn

Install :code:`nodejs`. This is only required for creating interactive molecule visualizations in Jupyter::

   conda install nodejs

Install the :code:`jupyterlab_3dmol` extension for visualizing molecules interactively in Jupyter::

   jupyter labextension install jupyterlab_3dmol

You should now be able to use the analysis components of conformer-rl for generating figures and visualizing molecule in Jupyter. To test that the installation was succesful, try running the example Jupyter notebook::

   jupyter-lab examples/example_analysis.ipynb


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
