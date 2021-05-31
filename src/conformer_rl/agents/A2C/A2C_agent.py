"""
A2C_agent
=========
"""
import torch
import torch.nn as nn

from conformer_rl.agents.base_ac_agent import BaseACAgent
from conformer_rl.utils import to_np
from conformer_rl.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent(BaseACAgent):
    """ Implements agent that uses the A2C (advantage actor critic) [1]_ algorithm.

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
    def _train(self) -> None:
        storage = self.storage
        config = self.config

        actions = storage.order('a')
        log_prob = storage.order('log_pi_a')
        value = storage.order('v')
        entropy = storage.order('ent')
        returns = torch.stack(self.returns, 1).view(config.num_workers * config.rollout_length, -1)
        advantages = torch.stack(self.advantages, 1).view(config.num_workers * config.rollout_length, -1)

        entropy_loss = entropy.mean()
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5*(returns - value).pow(2).mean()
        loss = policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss

        self.train_logger.add_scalar('policy_loss', policy_loss, self.total_steps)
        self.train_logger.add_scalar('value_loss', value_loss, self.total_steps)
        self.train_logger.add_scalar('entropy_loss', entropy_loss, self.total_steps)
        self.train_logger.add_scalar('loss', loss, self.total_steps)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.train_logger.add_scalar('grad_norm', grad_norm, self.total_steps)
        self.optimizer.step()
