"""
PPO_recurrent_agent
===================
"""
import numpy as np
import torch
import torch.nn as nn


from conformer_rl.agents.base_ac_agent_recurrent import BaseACAgentRecurrent
from conformer_rl.config import Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPORecurrentAgent(BaseACAgentRecurrent):
    """ Implements agent that uses the PPO (proximal policy optimization) [1]_ algorithm
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
    def __init__(self, config: Config):
        super().__init__(config)
        assert config.rollout_length % self.recurrence == 0
        assert config.mini_batch_size % self.recurrence == 0

    def _train(self) -> None:
        config = self.config
        storage = self.storage

        actions = storage.order('a')
        log_probs_old = storage.order('log_pi_a')
        returns = torch.stack(self.returns, 1).view(config.num_workers * config.rollout_length, -1)
        advantages = torch.stack(self.advantages, 1).view(config.num_workers * config.rollout_length, -1)

        recurrent_states = [storage.order(f'recurrent_states_{i}') for i in range(self.num_recurrent_units)]
        states = storage.order('states')

        self.train_logger.add_scalar('advantages', advantages.mean(), self.total_steps)
        advantages = (advantages - advantages.mean()) / advantages.std()

        ############################################################################################
        #Training Loop
        ############################################################################################
        for _ in range(config.optimization_epochs):
            indices = np.arange(0, self.config.rollout_length * self.config.num_workers, self.recurrence)
            indices = np.random.permutation(indices)

            num_indices = config.mini_batch_size // self.recurrence
            starting_batch_indices = [indices[i:i+num_indices] for i in range(0, len(indices), num_indices)]
            for starting_indices in starting_batch_indices:
                batch_entropy = 0
                batch_value_loss = 0
                batch_policy_loss = 0
                batch_loss = 0

                sampled_recurrent_states = (recurrent_states[i][:, starting_indices]for i in range(self.num_recurrent_units))

                for i in range(self.recurrence):
                    sampled_actions = actions[starting_indices + i]
                    sampled_log_probs_old = log_probs_old[starting_indices + i]
                    sampled_returns = returns[starting_indices + i]
                    sampled_advantages = advantages[starting_indices + i]

                    sampled_states = [states[j] for j in (starting_indices + i)]

                    prediction, sampled_recurrent_states = self.network(sampled_states, sampled_recurrent_states, sampled_actions)


                    entropy = prediction['ent'].mean()

                    ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                    obj = ratio * sampled_advantages
                    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                            1.0 + self.config.ppo_ratio_clip) * sampled_advantages

                    policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * entropy

                    value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                    loss = policy_loss + config.value_loss_weight * value_loss

                    batch_entropy += entropy.item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    if i < self.recurrence - 1:
                        for rstate_id, rstate in enumerate(recurrent_states):
                            rstate[:, starting_indices + i + 1] = sampled_recurrent_states[rstate_id].detach()


                batch_entropy /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                self.train_logger.add_scalar('entropy_loss', batch_entropy, self.total_steps)
                self.train_logger.add_scalar('policy_loss', batch_policy_loss, self.total_steps)
                self.train_logger.add_scalar('value_loss', batch_value_loss, self.total_steps)
                self.train_logger.add_scalar('loss', batch_loss, self.total_steps)

                self.optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()