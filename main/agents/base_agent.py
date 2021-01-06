import torch
import numpy as np
import time

import time
import torch
from ..utils import to_np

class BaseAgent:
    def __init__(self, config):
        self.config = config
        # self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def run_steps(self):
        config = self.config
        agent_name = self.__class__.__name__
        t0 = time.time()

        while True:
            if (config.save_interval != 0) and (self.total_steps % config.save_interval == 0):
                self.save('data/%s-%s-%d' % (agent_name, config.tag, self.total_steps))
            if (config.log_interval != 0) and (self.total_steps % config.log_interval == 0):
                # self.logger.info('steps %d, %.2f steps/s' % (self.total_steps, config.log_interval / (time.time() - t0)))
                t0 = time.time()
            if (config.eval_interval != 0) and not (self.total_steps % config.eval_interval == 0):
                self.eval_episodes()
            if (config.max_steps != 0) and (self.total_steps >= config.max_steps):
                self.close()
                break
            self.step()

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        raise NotImplementedError

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))