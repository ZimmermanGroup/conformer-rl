"""
A2C_recurrent_agent
===================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conformer_rl.agents.base_ac_agent_recurrent import BaseACAgentRecurrent
from conformer_rl.utils import to_np
from conformer_rl.agents.storage import Storage
from conformer_rl.config import Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CRecurrentAgent(BaseACAgentRecurrent):
    """ Implements agent that uses the A2C (advantage actor critic) [1]_ algorithm
    with support for recurrent neural networks.

    Parameters
    ----------
    config : :class:`~conformer_rl.config.agent_config.Config`
        Configuration object for the agent. See notes for a list of config
        parameters used by this agent.

    Notes
    -----
    *Config parameters:* The following parameters are required in the `config` object. See :class:`~conformer_rl.config.agent_config.Config`
    for more details on the parameters.

    * tag
    * train_env
    * eval_env
    * optimizer_fn
    * network
    * num_workers
    * rollout_length
    * max_steps
    * save_interval
    * eval_interval
    * eval_episodes
    * recurrence
    * discount
    * use_gae
    * gae_lambda
    * entropy_weight
    * value_loss_coefficient
    * gradient_clip
    * ppo_ratio_clip
    * data_dir
    * use_tensorboard

    *Logged values*: The following values are logged during training:
    
    * advantages
    * loss
    * policy_loss
    * entropy_loss
    * value_loss
    * episodic_return_eval (total rewards per episode for eval episodes)
    * episodic_return_train (total rewards per episode for training episodes)

    References
    ----------
    .. [1] `A2C Paper <https://arxiv.org/abs/1602.01783>`_

    """
    def __init__(self, config: Config):
        super().__init__(config)
        assert config.rollout_length * config.num_workers % self.recurrence == 0

    def _train(self) -> None:
        config = self.config
        storage = self.storage

        actions = storage.order('a')
        returns = torch.stack(self.returns, 1).view(config.num_workers * config.rollout_length, -1)
        advantages = torch.stack(self.advantages, 1).view(config.num_workers * config.rollout_length, -1)

        recurrent_states = [storage.order(f'recurrent_states_{i}') for i in range(self.num_recurrent_units)]
        states = storage.order('states')

        total_entropy, total_value_loss, total_policy_loss, total_loss = 0, 0, 0, 0

        starting_indices = np.arange(0, self.config.rollout_length * self.config.num_workers, self.recurrence)
        sampled_recurrent_states = (recurrent_states[i][:, starting_indices]for i in range(self.num_recurrent_units))
        for i in range(self.recurrence):
            sampled_actions = actions[starting_indices + i]
            sampled_states = [states[j] for j in (starting_indices + i)]
            sampled_returns = returns[starting_indices + i]
            sampled_advantages = advantages[starting_indices + i]

            prediction, sampled_recurrent_states = self.network(sampled_states, sampled_recurrent_states, sampled_actions)
            entropy_loss = prediction['ent'].mean()
            policy_loss = -(prediction['log_pi_a'] * sampled_advantages).mean()
            value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

            loss = policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss

            total_entropy += entropy_loss
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_loss += loss


        total_entropy /= self.recurrence
        total_policy_loss /= self.recurrence
        total_value_loss /= self.recurrence
        total_loss /= self.recurrence

        self.train_logger.add_scalar('policy_loss', total_policy_loss, self.total_steps)
        self.train_logger.add_scalar('value_loss', total_value_loss, self.total_steps)
        self.train_logger.add_scalar('entropy_loss', total_entropy, self.total_steps)
        self.train_logger.add_scalar('loss', total_loss, self.total_steps)


        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.train_logger.add_scalar('grad_norm', grad_norm, self.total_steps)
        self.optimizer.step()