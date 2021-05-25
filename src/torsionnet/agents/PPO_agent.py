import numpy as np
import torch
import torch.nn as nn

import time

from torsionnet.agents.base_agent import BaseAgent
from torsionnet.utils import to_np
from torsionnet.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent(BaseAgent):
    def step(self):
        self.storage.reset()
        with torch.no_grad():
            self._sample()
        self._calculate_advantages()
        self._train()
        
    def _train(self):
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

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()