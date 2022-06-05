Utilizing Curriculum Learning
=============================
This section walks through how to train an agent using curriculum learning.

What is Curriculum Learning?
----------------------------
Curriculum learning can be viewed as applying transfer learning iteratively. In order to train an agent on a specific task, the agent
is first on a similar but easier task. Once the agent has learned that task, it will then be trained on a slightly more difficult task. This continues until the agent is trained directly on the original task.

Previous empirical results have shown that through curriculum learning, an agent can learn difficult tasks that it is not able to learn by training directly on the task itself. Even if it is able to learn a task by training directly on that task, curriculum learning often makes the training process more efficient (it reduces the training time required).

:mod:`conformer_rl` contains implementations of mixin classes that can make any of the included environments and agents compatible with curriculum learning.

Curriculum Learning Example Training Script
-------------------------------------------
The full code for this example can be found in `examples/curriculum_example.py <https://github.com/ZimmermanGroup/conformer-rl/blob/master/examples/curriculum_example.py>`_.

In this example, we want to train an agent to generate conformers for a branched alkane molecule with 16 carbon atoms. However, instead of training directly on this molecule, we will utilize a curriculum where the agent begins by training on a branched alkane with 8 atoms, and then iteratively moves up to a branched alkane with 15 atoms.

We first generate the :class:`~conformer_rl.config.mol_config.MolConfig` objects for the training and evaluation environments. For the training environment, we want a list of :class:`~conformer_rl.config.mol_config.MolConfig` objects starting with a branched alkane with 8 carbon atoms, up to a branched alkane with 15 carbon atoms::

    # Create mol_configs for the curriculum
    mol_configs = [config_from_rdkit(generate_branched_alkane(i), num_conformers=200, calc_normalizers=True) for i in range(8, 16)]

Next, we create a mol_config for the evaluation environment. Note that the evaluation environment will not be a curriculum environment since we are only evaluating the agent on a single conformer::

    eval_mol_config = config_from_rdkit(generate_branched_alkane(16), num_conformers=200, calc_normalizers=True)

Next, we will set up the :class:`~conformer_rl.config.agent_config.Config` object for the agent and hyperparameters as we have done in the previous sections::

    config = Config()
    config.tag = 'curriculum_test'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)

    # Batch Hyperparameters
    config.max_steps = 100000

    # training Hyperparameters
    lr = 5e-6 * np.sqrt(10)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # curriculum Hyperparameters
    config.curriculum_agent_buffer_len = 20
    config.curriculum_agent_reward_thresh = 0.7
    config.curriculum_agent_success_rate = 0.7
    config.curriculum_agent_fail_rate = 0.2

We will now create the environments for training and evaluation. :mod:`conformer_rl` already has pre-built environments for curriculum learning. We will use the :class:`~conformer_rl.environments.environments.GibbsScorePruningCurriculumEnv` environment which is the same as the :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv` we used previously except it is now compatible with curriculum learning. We will set the evaluation env to :class:`~conformer_rl.environments.environments.GibbsScorePruningEnv`::

    # Task Settings
    config.train_env = Task('GibbsScorePruningCurriculumEnv-v0', concurrency=True, num_envs=10, seed=np.random.randint(0,1e5), mol_configs=mol_configs)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=eval_mol_config)
    config.eval_interval = 20000

Next, we need to specify hyperaparameters specific to the curriculum. The specific meaning of each hyperparameter is discussed in :ref:`Curriculum-Supported Agents` and :ref:`Curriculum Conformer_env`::

    # curriculum Hyperparameters
    config.curriculum_agent_buffer_len = 20
    config.curriculum_agent_reward_thresh = 0.7
    config.curriculum_agent_success_rate = 0.7
    config.curriculum_agent_fail_rate = 0.2

Finally, we initiate our agent. Each of the pre-built agents in :mod:`conformer_rl` has a curriculum version as well. In this example we will use :class:`~conformer_rl.agents.curriculum_agents.PPORecurrentExternalCurriculumAgent`::

    agent = PPORecurrentExternalCurriculumAgent(config)
    agent.run_steps()

We can now run the script to train the agent.

For more information on how the curriculum environments and agents work, see the sections :ref:`Curriculum Conformer_env` and :ref:`Curriculum-Supported Agents`.


