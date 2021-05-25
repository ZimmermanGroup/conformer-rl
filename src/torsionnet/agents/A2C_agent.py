import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torsionnet.agents.base_agent import BaseAgent
from torsionnet.utils import to_np
from torsionnet.agents.storage import Storage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent(BaseAgent):
    def _train(self):
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

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.train_logger.add_scalar('grad_norm', grad_norm, self.total_steps)
        self.optimizer.step()
