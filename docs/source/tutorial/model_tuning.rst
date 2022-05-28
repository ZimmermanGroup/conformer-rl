Hyperparameter Tuning and Additional Options
============================================

This section walks through more advanced options and configurations when training an agent, such as specifying the neural network to be used, setting hyperparameters, and more.

The code in this section will follow the training script `examples/example1.py <https://github.com/ZimmermanGroup/conformer-rl/blob/master/examples/example2.py>`_.

Note that the options discussed here only covers a subset of all the possible options available when training an agent, and more options may be added in the future. For an updated full list of configurable options, see the attributes for the :class:`~conformer_rl.config.agent_config.Config` object.

Configuring Molecule and Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As in :ref:`Getting Started - Training a Conformer Generation Agent`, we set up the molecule and environment. In this example, we generate conformers for a lignin polymer with three monomers::
    # configure molecule
    mol = generate_lignin(3)
    mol_config = config_from_rdkit(mol, calc_normalizers=True, save_file='lignin')

    # create agent config and set environment
    config = Config()
    config.tag = 'example2'
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=20, mol_config=mol_config, max_steps=200)

Configuring the Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:mod:`conformer_rl` contains implementations of several graph neural network models, which can be found in :ref:`models`. One neural network architecture that has performed well empirically in conformer generation is :class:`~conformer_rl.models.RTGN_recurrent.RTGNRecurrent`, which we will use in this example::
    config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(device)
    
Notice that the observation from :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv`
is a graph where each node embedding has a dimension of 5 and each edge embedding has a dimension of 6, 
so we must specify ``edge_dim=6`` and ``node_dim=5`` when initializing the neural network.

Configuring Logging
^^^^^^^^^^^^^^^^^^^
Next, we configure logging options::

    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

The first option specifies that the trained neural network parameters will be saved every 20,000 steps. The saved neural network parameters can be used for evaluation in dowstream tasks. The second option specifies that logs (for Tensorboard, model performance on the evaluation environment, and saved neural network parameters) will be saved in a directory called ``data``. The final option enables tensorboard logging, so we can track agent training progress.

Configuring the Evaluation Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we can set up evaluation of the agent. In this example, we will have the agent be evaluated every 20000 steps, and we will set the eval environment to be conformer generation for a lignin polymer with four monomers (instead of three). Thus, the evaluation environment will allow us to see whether the agent is able to generalize from three monomer lignin to four monomer lignin. We will also have the agent evaluate for 2 episodes during each evaluation::

    eval_mol = generate_lignin(4)
    eval_mol_config = config_from_rdkit(mol, calc_normalizers=True, ep_steps=200, save_file='lignin_eval')
    config.eval_env = Task('GibbsScorePruningEnv-v0', num_envs=1, mol_config=eval_mol_config, max_steps=200)
    config.eval_interval = 20000
    config.eval_episodes = 2

Tuning Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^
Finally, we can set the other hyperparameters. For more information on what each of the hyperparameters represent, see the API reference for :class:`~conformer_rl.config.agent_config.Config`::
    
    # Batch Hyperparameters
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 200000
    config.mini_batch_size = 50

    # Training Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

Running the Agent
^^^^^^^^^^^^^^^^^
We can then create and train the agent. We will use the PPO algorithm, so we will use :class:`~conformer_rl.agents.PPO.PPO_agent.PPOAgent`::

    agent = PPOAgent(config)
    agent.run_steps()

Viewing Results
^^^^^^^^^^^^^^^

After training the agent, we can view the training metrics and track training progress using Tensorboard::

    $ tensorboard --logdir data/tensorboard_log/

