Installation
============

Since :mod:`conformer_rl` can be run within a Conda environment, it should work on all platforms (Windows, MacOS, and Linux).

Prerequisites
-------------
* We recommend installing in a new Conda environment.

   * If you are new to using Conda, you can install it `here <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ and learn more about environments `here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

* Install dependencies

   * Install `PyTorch <https://pytorch.org/>`_ . PyTorch version of 1.8.0 or greater is required for :mod:`conformer_rl`.
   * Install `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ .

      * **Important Note**: Please make sure to use the same package installer (i.e., Conda, Pip) for installing both PyTorch and PyTorch geometric.

   * Install `RDKit <https://www.rdkit.org/>`_::

      conda install -c conda-forge rdkit

* Finally, install :mod:`conformer_rl`::

   pip install conformer-rl

Verify Installation
-------------------
As a quick check to verify the installation has succeeded, navigate to the ``examples`` directory
and run `examples/test_example.py <https://github.com/ZimmermanGroup/conformer-rl/blob/master/examples/test_example.py>`_. The script should finish running in a few minutes or less. If no errors are encountered
then most likely the installation has succeeded.

Additional Installation for Analysis/Visualization Tools
--------------------------------------------------------
Some additional dependencies are required for visualizing molecules in Jupyter/IPython notebooks.

Firstly, install the dependencies required for visualizing molecules and figures in Jupyter (these should already be installed after installing conformer-rl)::

   pip install conformer-rl[visualization]

Install :code:`nodejs`. This is only required for creating interactive molecule visualizations in Jupyter::

   conda install nodejs

Install the :code:`jupyterlab_3dmol` extension for visualizing molecules interactively in Jupyter::

   jupyter labextension install jupyterlab_3dmol

You should now be able to use the analysis components of conformer-rl for generating figures and visualizing molecule in Jupyter. To test that the installation was succesful, try running the example Jupyter notebook::

   jupyter-lab examples/example_analysis.ipynb

Additional Installation for Molecule Generation Tools
-----------------------------------------------------
Additional dependencies are required for generating non-alkane molecules such as lignin. To install these dependencies run::
    
    pip install conformer-rl[generate_molecules]

A full list of classes of molecules that :mod:`conformer_rl` can generate can be found in :ref:`Molecule Generation`.