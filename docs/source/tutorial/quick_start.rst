Quick Start
===========

This section walks through how to write a training script
using the pre-built environments and agents in :mod:`conformer_rl`.

Several example scripts are available in the `examples <https://github.com/ZimmermanGroup/conformer-rl/tree/master/examples>`_ directory.

Creating a training script
--------------------------
The full code for this example can be found at
`examples/basic_example/run.py <https://github.com/ZimmermanGroup/conformer-rl/blob/master/examples/basic_example/run.py>`_.

Suppose we want to train an agent to generate low-energy conformers for
a branched alkane molecule with 18 carbon atoms.

Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we choose one of the pre-built environments for simulating conformer generation.
We can configure an environment with a branched alkane
with 18 carbon atoms. :mod:`conformer_rl` has a function :func:`~conformer_rl.molecule_generation.molecules.branched_alkane` 
that automatically generates the environment configuration
:Class:`~conformer_rl.config.mol_config.MolConfig` object for branched alkanes::

    from conformer_rl.molecule_generation import branched_alkane
    alkane_env_config = branched_alkane(num_atoms=18)

Next, we can set up the environment that the agent will train on.
We will choose :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv` since
it penalizes conformers that are identical to previously generated conformers, encouraging the
agent to find a diverse set of conformers. It is a pre-built environment that is already
registered with OpenAI gym as ``'GibbsScorePruningEnv-v0'``. We can also specify how many conformers to be generated
in each episode of the environment by setting ``max_steps``.

:func:`~conformer_rl.environments.environment_wrapper.Task` automatically generates
an environment wrapper compatible with the agent. By setting ``num_envs`` to 20, we can have our agent
sample on 20 environments simultaneously, which can speed up training.::

    from conformer_rl.environments import Task
    training_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=20, mol_config=alkane_env_config, max_steps=200)

Configuring the agent
^^^^^^^^^^^^^^^^^^^^^
We will now configure the agent. Agents are initialized and configured with
the :class:`~conformer_rl.config.agent_config.Config` object, which specifies parameters
and hyperparameters for the agent. We begin by creating a config object and adding in information
about our training envrionment::

    from conformer_rl.config import Config
    config = Config()
    config.tag = 'tutorial'
    config.train_env = training_env

Next, we can specify the agent details. Suppose we want to use the PPO algorithm, and we want
to use the pre-built :class:`~conformer_rl.models.RTGN_recurrent.RTGNRecurrent` neural network
for prediction. We can specify this as well. Notice that the observation from :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv`
is a graph where each node embedding has a dimension of 5 and each edge embedding has a dimension of 6, 
so we must specify this when initializing the neural network::

    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

Since we are running 20 environments in parallel, we must set::

    config.num_workers = 20

We want to log training metrics to Tensorboard, save neural network parameters every 20000 steps,
and save the logs in a directory called ``data``,
so we set the following::

    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

Next, we can set up evaluation of the agent. If evaluation is enabled, the agent
will be evaluated on an eval environment. The eval environment can be the same or different
as the training environment. Results from the eval environment will be saved as .pickle files
in subdirectories of ``data/env_data``, and can be analyzed/visualized using the :mod:`~conformer_rl.analysis.analysis`
module. An example of using the :mod:`~conformer_rl.analysis.analysis` module can be found in
`examples/example_analysis.ipynb <https://github.com/ZimmermanGroup/conformer-rl/blob/master/examples/example_analysis.ipynb>`_.

In this example, we will have the agent be evaluated every 20000 steps, and we will set the
eval environment to be the same as the training environment. We will also have the agent evaluate for
2 episodes during each evaluation::

    config.eval_env = Task('GibbsScorePruningEnv-v0', num_envs=1, mol_config=alkane_env_config, max_steps=200)
    config.eval_interval = 20000
    config.eval_episodes = 2

Tuning hyperparameters
^^^^^^^^^^^^^^^^^^^^^^

Finally, we can set the other hyperparmeters. For more information on what each of
the hyperparameters represent, see the API reference for :class:`~conformer_rl.config.agent_config.Config`::
    
    # Batch Hyperparameters
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 10000000
    config.mini_batch_size = 50

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

Running the agent
^^^^^^^^^^^^^^^^^

We can then create and train the agent. Since we want to use the PPO algorithm, and our neural network
utilizes recurrent states, we will use :class:`~conformer_rl.agents.PPO.PPO_recurrent_agent.PPORecurrentAgent`::

    from conformer_rl.agents import PPORecurrentAgent
    agent = PPORecurrentAgent(config)
    agent.run_steps()

Viewing results
^^^^^^^^^^^^^^^

After training the agent, we can view the training metrics and track training progress using Tensorboard::

    $ tensorboard --logdir data/tensorboard_log/

