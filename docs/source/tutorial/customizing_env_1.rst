Customizing Environment - Part One
==================================

This section will cover the different components of a :mod:`conformer_rl` environment and a basic example of how to customize the environment from pre-built environment components.

ConformerEnv and Environment Components
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
  which returns an observation object based on the current configuration of the molecule and is a compatible input for the neural net being used for training.

Other methods and functionality can also be added, but the above three components are the main ones and
cover most of the functionality of any environment.

Creating and registering new environments
-----------------------------------------

The source code for this experiment can be found in `examples/custom_env_basic <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples/custom_env_basic>`_. The code for setting up the new environment is found in ``custom_env.py``. The code for the updated training script is found in ``run.py``.

In :ref:`Getting Started - Training a Conformer Generation Agent` we trained an agent on one of the pre-built environments, :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv`. Notice that the observation handler for :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv` creates an embedding for each node/atom that includes both a one-hot representation for whether the atom is a carbon or oxygen atom, as well as the x, y, z coordinates for the atom resulting in a vector of dimension 5 for each node.

Suppose we wanted to see whether having the x, y, and z coordinates in the graph representation of the molecule is useful for the task of generating conformers. To test this, we can create an environment where the observation handler returns a graph that only contains the one-hot representation for whether the atom is a carbon or oxygen for each atom, and does not contain the x, y, z positional information.

:mod:`conformer_rl` already has an implementation of this type of observation handler: :class:`~conformer_rl.environments.environment_components.obs_mixins.AtomTypeGraphObsMixin`.

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

We can try training an agent on this new environment by modifying the training script in :ref:`Getting Started - Training a Conformer Generation Agent` and see if the results have changed. The full training script code for this example can be found in `examples/custom_env_basic/run.py <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples/custom_env_basic/run.py>`_. First, import the ``custom_env.py`` module to run
the gym registration code::

  # run.py
  # import the custom created environment to run the gym register script
  import custom_env

The setup for the molecule config will be the same as in :ref:`Getting Started - Training a Conformer Generation Agent`, so we will not explicitly cover the details here.
Next, we should change the tag of the agent to represent the environment of this experiment::

  # run.py
  # set the tag to reflect the custom environment
  config.tag = 'atom_type_env'

Additionally, since each node of the graph returned by the observation handler now has a dimension of only 2,
we must initialize the neural network with the correct ``node_dim``. In :ref:`Getting Started - Training a Conformer Generation Agent`, we did not explicitly set the neural network, so the neural network was set by default to :class:`~conformer_rl.models.RTGN_recurrent.RTGNRecurrent`. In this example, we will use the same network and initialize it with the correct ``node_dim``::

  # run.py
  # Update the network's node_dim to equal 2
  config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=2).to(device)

Finally, when setting the ``train_env`` and ``eval_env``, we must specify the name of the environment to be the ``'Test-Env-v0'`` we registered::
  
  # Set the environment to the test env
  config.train_env = Task('TestEnv-v0', concurrency=True, num_envs=5, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=200)
  config.eval_env = Task('TestEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)

  