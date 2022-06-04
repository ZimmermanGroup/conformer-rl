"""
Curriculum-Supported Agents
===========================
"""

import logging
import time
from collections import deque

import numpy as np
import torch

class ExternalCurriculumAgentMixin():
    """General mixin class to enable curriculum

    Adds functionality to an existing agent for externally interacting with an environment supporting curriculum learning.


    Parameters
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object for the agent. See notes for a list of config
        parameters used by this agent.

    Notes
    -----
    In addition to the config parameters required for the base agent class, use of this mixin
    requires the following additional parameters in the :class:`~conformer_rl.config.agent_config.Config` object:

    * curriculum_agent_buffer_len
    * curriculum_agent_reward_thresh
    * curriculum_agent_success_rate
    * curriculum_agent_fail_rate
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_buffer = deque([], maxlen=config.curriculum_agent_buffer_len)
        self.curriculum_buffer_len = config.curriculum_agent_buffer_len
        self.curriculum_reward_thresh = config.curriculum_agent_reward_thresh
        self.curriculum_success_rate = config.curriculum_agent_success_rate
        self.curriculum_fail_rate = config.curriculum_agent_fail_rate

    def step(self) -> None:
        """Performs one iteration of acquiring samples on the environment
        and then trains on the acquired samples.
        """
        # sample 
        self.storage.reset()
        sample_start = time.time()
        self._sample()
        logging.debug(f'sample time: {time.time() - sample_start} seconds')

        # update curriculum
        self.update_curriculum()

        # train
        train_start = time.time()
        self._calculate_advantages()
        self._train()
        logging.debug(f'train time: {time.time() - train_start} seconds')

    def update_curriculum(self) -> None:
        """Evaluates the current performance of the agent and signals the environment to
        increase the level (difficulty) or decrease it depending on the agent's performance.

        The agent is evaluated only when the number of episodes elapsed since the last evaluation
        has exceeded the parameter ``curriculum_agent_buffer_len`` assigned in the :class:`~conformer_rl.config.agent_config.Config` object.
        During the evaluation, the ratio of episodes (out of the last ``curriculum_agent_buffer_len`` episodes) which have a reward exceeding
        the ``curriculum_agent_reward_thresh`` parameter defined in the :class:`~conformer_rl.config.agent_config.Config`
        is calculated. If this ratio exceeds the ``curriculum_agent_success_rate`` parameter, the environment is signaled
        to increase the difficulty of the curriculum. This is done by calling the ``increase_level`` method of the environment.
        If the ratio is less than the ``curriculum_agent_fail_rate`` parameter, the environment is told to decrease the difficulty.
        """
        current_terminals = torch.cat(self.storage['terminals']).squeeze()
        current_rewards = torch.cat(self.storage['r']).squeeze()
        self.reward_buffer.extend(current_rewards[current_terminals == True].tolist())

        if len(self.reward_buffer) >= self.curriculum_buffer_len:
            rewbuf = np.array(self.reward_buffer)
            pass_rate = (rewbuf >= self.curriculum_reward_thresh).mean()

            if pass_rate > self.curriculum_success_rate:
                self.task.env_method('increase_level')
                self.reward_buffer.clear()
            elif pass_rate < self.curriculum_fail_rate:
                self.task.env_method('decrease_level')
                self.reward_buffer.clear()

from conformer_rl.agents import PPOAgent, PPORecurrentAgent
from conformer_rl.agents import A2CAgent, A2CRecurrentAgent

class PPOExternalCurriculumAgent(ExternalCurriculumAgentMixin, PPOAgent):
    """Implementation of :mod:`~conformer_rl.agents.PPO.PPO_agent.PPOAgent` compatible with
    environments that use curriculum learning. See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum`
    for more details.
    
    """
    pass

class PPORecurrentExternalCurriculumAgent(ExternalCurriculumAgentMixin, PPORecurrentAgent):
    """Implementation of :mod:`~conformer_rl.agents.PPO.PPO_recurrent_agent.PPORecurrentAgent` compatible with
    environments that use curriculum learning. See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum`
    for more details.
    
    """
    pass

class A2CExternalCurriculumAgent(ExternalCurriculumAgentMixin, A2CAgent):
    """Implementation of :mod:`~conformer_rl.agents.A2C.A2C_agent.A2CAgent` compatible with
    environments that use curriculum learning. See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum`
    for more details.
    
    """
    pass

class A2CRecurrentExternalCurriculumAgent(ExternalCurriculumAgentMixin, A2CRecurrentAgent):
    """Implementation of :mod:`~conformer_rl.agents.A2C.A2C_recurrent_agent.A2CRecurrentAgent` compatible with
    environments that use curriculum learning. See :meth:`~conformer_rl.agents.curriculum_agents.ExternalCurriculumAgentMixin.update_curriculum`
    for more details.
    
    """
    pass