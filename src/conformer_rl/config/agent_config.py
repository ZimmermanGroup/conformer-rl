"""
Agent_config
============
"""

import torch
from conformer_rl.models import RTGNGatRecurrent

class Config:
    """Configuration object for agents.

    Specifies parameters and hyperparameters for agents. See attributes below for details.

    Attributes
    ----------
    tag : str, required for all agents
        Used to identify the training run in saved log files and in Tensorboard.
    train_env : wrapper for environments from :func:`~conformer_env.environments.environment_wrapper.Task`, required for all agents
        Wrapper for environments used to train the agents.
    eval_env : wrapper for environments from :func:`~conformer_env.environments.environment_wrapper.Task`, optional
        Wrapper for environment used to evaluate the agent.
    network : pytorch neural network module (torch.nn.Module), required for all agents
        Neural network to be used by the agent.
    optimizer_fn : lambda(iterable) -> torch.optim.Optimizer, required for all agents
        Lambda function that maps the parameters of a torch.nn.module 
        (as obtained by calling the `.parameters()` method on the module) to a torch.optim.Optimizer function
        by passing in the parameters to the constructor of the optimizer function. For example::
        
            config.optimizer_fn = lambda params : torch.optim.Adam(params, lr=0.001)

    num_workers : int, required by all agents
        Number of parallel environments to sample from during training.
    rollout_length : int, required by all agents
        Number of environment steps taken by each worker during each sampling iteration.
    max_steps : int, required by all agents
        Number of environment steps to take before ending agent training.
    save_interval : int, required by all agents
        How often (in environment steps) to save neural network parameters. If set to 0,
        parameters will not be saved.
    eval_interval : int, required by all agents
        How often to evaluate the agent on the eval environment.
    eval_episodes : int, required by all agents
        How many episodes to evaluate the agent during each evaluation.
    recurrence : int, required by recurrent agents
        Number of steps taken before resetting recurrent states when training agent/updating network weights.
    optimization_epochs : int
        Number of epochs for training each minibatch. Used for PPO and PPORecurrent agents.
    mini_batch_size : int
        Size of each mini batch to train on. Used for PPO and PPORecurrent agents.

    discount : float, required by all agents.
        Discount factor (often denoted by γ) used for advantage estimation.
    use_gae : bool, required by all agents.
        Determines whether to use generalized advantage estimation (GAE) for estimating advantages, or
        SARSA update.
    gae_lambda : float, required by all agents if `use_gae` is ``True``
        The λ parameter used by the generalized advantage estimator (gae). See [1]_ for details.
    entropy_weight : float, required by all agents
        Coefficient for the entropy when calculating total loss.
    value_loss_coefficient : float, required by all agents
        Coefficient for the value loss when calculating total loss.
    gradient_clip : float, required by all agents
        Max norm for clipping gradients for neural network.
    ppo_ratio_clip : float, required by PPO and PPORecurrent agents.
        Clipping parameter ε for PPO algorithm, see [2]_ for details.

    curriculum_agent_buffer_len : int, required by all curriculum agents
        The number of most recent completed episodes in which to evaluate the agent on for curriculum learning.
        See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum` for more details on how
        curriculum learning is implemented.
    curriculum_agent_reward_thresh : float, required by all curriculum agents
        The reward threshold for considering the agent to have "succeeded" in an episode. Used for evaluating the agent
        for curriculum learning.
        See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum` for more details on how
        curriculum learning is implemented.
    curriculum_agent_success_rate : float, required by all curriculum agents
        The minimum success rate for the agent to signal the environment to increase the level/difficulty for the curriculum.
        See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum` for more details on how
        curriculum learning is implemented.
    curriculum_agent_fail_rate : float, required by all curriculum agents
        The maximum success rate for the agent to signal the environment to decrease the level/difficulty of the curriculum.
        See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum` for more details on how
        curriculum learning is implemented.

    data_dir : str, required by all agents
        Directory path for saving log files.
    use_tensorboard : bool, required by all agents
        Whether or not to save agent information to Tensorboard.


    References
    ----------
    .. [1] `Generalized Advantage Estimation (GAE) paper <https://arxiv.org/abs/1506.02438>`_
    .. [2] `PPO Paper <https://arxiv.org/abs/1707.06347>`_
    

    """
    def __init__(self):

        # naming
        self.tag = 'conformer_generation'

        # training objects
        self.train_env = None
        self.eval_env = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = RTGNGatRecurrent(6, 128, node_dim=5).to(self.device)
        self.optimizer_fn = lambda params : torch.optim.Adam(params, lr=1e-5, eps=1e-5)

        # batch hyperparameters
        self.rollout_length = 20
        self.max_steps = 50000
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 1
        self.recurrence = 2
        self.optimization_epochs = 4
        self.mini_batch_size = 24

        # training hyperparameters
        self.discount = 0.9999
        self.use_gae = True
        self.gae_lambda = 0.95
        self.entropy_weight = 0.001
        self.value_loss_weight = 0.25
        self.gradient_clip = 0.5
        self.ppo_ratio_clip = 0.2

        # curriculum hyperparameters
        self.curriculum_agent_buffer_len = 20
        self.curriculum_agent_reward_thresh = 0.7
        self.curriculum_agent_success_rate = 0.7
        self.curriculum_agent_fail_rate = 0.2

        # logging config
        self.data_dir = 'data'
        self.use_tensorboard = True













