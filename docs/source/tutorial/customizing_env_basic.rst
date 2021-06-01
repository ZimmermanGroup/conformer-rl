Customizing Environment - basic
===============================

This section will cover the different components of a :mod:`conformer_rl` environment
and a basic example of how to customize the environment from pre-built environment components.

ConformerEnv and environment components
---------------------------------------
Environments in :mod:`conformer_rl` inherit from the
:class:`~conformer_rl.environments.conformer_env.ConformerEnv`
interface. :class:`~conformer_rl.environments.conformer_env.ConformerEnv` is designed so that
different components of the environment are modularized and can be implemented independently from each other.

The main components of the :class:`~conformer_rl.environments.conformer_env.ConformerEnv` include

* **Action Handler** refers to overriding of the :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._step` method of
  :class:`~conformer_rl.environments.conformer_env.ConformerEnv`, which determines how the molecule is modified given some action.
* **Reward Handler** refers to overriding of the :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._reward` method of
  :class:`~conformer_rl.environments.conformer_env.ConformerEnv`, 
  which determines how the reward is calculated based on the current configuration of the molecule.
* **Observation Handler** refers to overriding of the :meth:`~conformer_rl.environments.conformer_env.ConformerEnv._obs` method of
  :class:`~conformer_rl.environments.conformer_env.ConformerEnv`, 
  which returns an observation object based on hte current configuration of the molecule and is a compatible input for the neural net being used for training.

Other methods and functionality can also be added, but the above three components are the main ones and
cover most of the functionality of any environment.

Creating and registering new environments
-----------------------------------------

The source code for this experiment can be found in `examples/custom_env_basic <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples/custom_env_basic>`_. The code for
the original pre-built environment is in ``run.py`` and the code for the new environment is found
in ``custom_env.py``. The code for the updated training script is found in ``custom_run.py``.

In :ref:`Quick Start` we trained an agent on one of the pre-built environments, :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv`.
Notice that the observation handler for :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv` creates an embedding for each node/atom
that includes both a one-hot representation for whether the atom is a carbon or oxygen atom, as well as the x, y, z coordinates for the atom resulting in a
vector of dimension 5 for each node.

Suppose we wanted to see whether having the x, y, and z coordinates in the graph representation of the molecule
is useful for the task of generating conformers. To test this, we can create an environment where the observation handler
returns a graph that only contains the one-hot representation for whether the atom is a carbon or oxygen for each atom, and does not
contain the x, y, z positional information.

:mod:`conformer_rl` already has an implementation of this type of observation handler:
:class:`~conformer_rl.environments.environment_components.obs_mixins.AtomTypeGraphObsMixin`.

Thus, we can use this mixin to create our custom environment class::

  # custom_env.py
  class TestEnv(DiscreteActionMixin, AtomTypeGraphObsMixin, GibbsPruningRewardMixin, ConformerEnv):
    pass

Next, since this is not a pre-built environment, we must register the environment with OpenAI gym::

  # custom_env.py
  # register the environment with OpenAI gym
  gym.register(
      id='TestEnv-v0',
      entry_point='custom_env:TestEnv'
  )

We can try training an agent using the same parameters as in :ref:`Quick Start` and
see if the results have changed. Make sure to import the ``custom_env.py`` module to run
the gym registration code::

  # custom_run.py
  import custom_env

We should change the tag of the agent to represent
the details of this experiment::

  # custom_run.py
  # set the tag to represent the run
  config.tag = 'atom_type_test'

Additionally, since each node of the graph returned by the observation handler now has a dimension of only 2,
we must initialize the neural network with the correct `node_dim`::

  # custom_run.py
  # Update the network's node_dim to equal 2
  config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=2).to(device)

Finally, we can run the agent and compare the results on Tensorboard.

Custom molecules and :class:`~conformer_rl.config.mol_config.MolConfig`
-----------------------------------------------------------------------
In the last two examples, we have used :func:`~conformer_rl.molecule_generation.molecules.branched_alkane` to automatically generate
the :Class:`~conformer_rl.config.mol_config.MolConfig` object for configuring environments. This section shows how one can create a 
:Class:`~conformer_rl.config.mol_config.MolConfig` for a custom molecule.

The :Class:`~conformer_rl.config.mol_config.MolConfig` object is used similarly to the :Class:`~conformer_rl.config.agent_config.Config` object
for configuring agents, and has a lot less parameters. The main parameter is molecule itself. Suppose we have a custom rdkit molecule::

  from rdkit import Chem
  mol = Chem.MolFromSmiles('Cc1ccccc1')

To create a :Class:`~conformer_rl.config.mol_config.MolConfig` for this molecule we simply set the `mol` attribute to the molecule. However,
we should make sure to add hydrogens and sanitize the molecule first::

  from conformer_rl.config import MolConfig
  config = MolConfig()

  mol = Chem.AddHs(mol) # add hydrogens
  Chem.AllChem.MMFFSanitizeMolecule(mol) # sanitize molecule
  
  config.mol = mol

Additionally, if the environment utilizes the Gibbs Score reward [1]_,
the constants :math:`E_0` and :math:`Z_0` need to be calculated. :mod:`conformer_rl`
contains a function :func:`~conformer_rl.utils.chem_utils.calculate_normalizers` that
does this automatically::

  from conformer_rl.utils import calculate_normalizers
  E0, Z0 = calculate_normalizers(mol)

  config.E0 = E0
  config.Z0 = Z0

After that, `config` is complete and can be used with the pre-built environments in :mod:`conformer_rl`.


.. [1] `TorsionNet Paper <https://arxiv.org/abs/2006.07078>`_

  