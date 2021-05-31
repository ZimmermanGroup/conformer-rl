"""
PPO_agent
=========
"""
import numpy as np
import torch
import torch.nn as nn

import time

from conformer_rl.agents.base_ac_agent import BaseACAgent
from conformer_rl.utils import to_np
from conformer_rl.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent(BaseACAgent):
    """ Implements agent that uses the PPO (proximal policy optimization) [1]_ algorithm.

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
    * optimization_epochs
    * mini_batch_size
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
    .. [1] `PPO Paper <https://arxiv.org/abs/1707.06347>`_

    """
    def step(self) -> None:
        """Performs one iteration of acquiring samples on the environment
        and then trains on the acquired samples.
        """
        self.storage.reset()
        with torch.no_grad():
            self._sample()
        self._calculate_advantages()
        self._train()
        
    def _train(self) -> None:
        config = self.config
        storage = self.storage

        actions = storage.order('a')
        log_probs_old = storage.order('log_pi_a')
        returns = torch.stack(self.returns, 1).view(config.num_workers * config.rollout_length, -1)
        advantages = torch.stack(self.advantages, 1).view(config.num_workers * config.rollout_length, -1)
        states = storage.order('states')

        self.train_logger.add_scalar('advantages', advantages.mean(), self.total_steps)
        advantages = (advantages - advantages.mean()) / advantages.std()

        ############################################################################################
        #Training Loop
        ############################################################################################
        for _ in range(config.optimization_epochs):
            indices = np.arange(0, self.config.rollout_length * self.config.num_workers)
            indices = np.random.permutation(indices)

            num_indices = config.mini_batch_size
            starting_batch_indices = [indices[i:i+num_indices] for i in range(0, len(indices), num_indices)]
            for starting_indices in starting_batch_indices:

                sampled_actions = actions[starting_indices]
                sampled_log_probs_old = log_probs_old[starting_indices]
                sampled_returns = returns[starting_indices]
                sampled_advantages = advantages[starting_indices]

                sampled_states = [states[j] for j in (starting_indices)]

                prediction = self.network(sampled_states, sampled_actions)

                entropy = prediction['ent'].mean()
                prediction['log_pi_a'] = prediction['log_pi_a']
                prediction['v'] = prediction['v']

                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                        1.0 + self.config.ppo_ratio_clip) * sampled_advantages

                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * entropy

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                loss = policy_loss + config.value_loss_weight * value_loss

                self.train_logger.add_scalar('entropy_loss', entropy, self.total_steps)
                self.train_logger.add_scalar('policy_loss', policy_loss, self.total_steps)
                self.train_logger.add_scalar('value_loss', value_loss, self.total_steps)
                self.train_logger.add_scalar('loss', loss, self.total_steps)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()