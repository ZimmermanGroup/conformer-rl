import torch
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torsionnet.utils import *
from torsionnet.logging import EvalLogger

class BaseAgent:
    def __init__(self, config):
        self.config = config
        # self.writer = SummaryWriter(log_dir = f'./train_data/{config.tag}-{get_time_str()}')
        self.task_ind = 0
        # mkdir('train_data')
        # mkdir('model_data')
        # mkdir('molecule_data')

        self.eval_logger = EvalLogger(dir=f"./data/eval/{config.tag}-{get_time_str()}")

    def run_steps(self):
        config = self.config
        agent_name = self.__class__.__name__

        while True:
            if (config.save_interval != 0) and (self.total_steps % config.save_interval == 0):
                pass
                # self.save(f'model_data/{agent_name}-{config.tag}-{self.total_steps}')
            if (config.eval_interval != 0) and (self.total_steps % config.eval_interval == 0):
                self.eval_episodes()
            if (config.max_steps != 0) and (self.total_steps >= config.max_steps):
                self.close()
                break
            self.step()

    def close(self):
        self.task.close()

    def save(self, filename):
        pass
        # torch.save(self.network.state_dict(), '%s.model' % (filename))

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def eval_episode(self):
        raise NotImplementedError

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            self.eval_ep = ep
            total_rewards = self.eval_episode()
            episodic_returns.append(total_rewards)
            self.eval_logger.log_episode({"total_rewards": total_rewards})
            self.eval_logger.dump_episode(self.total_steps, self.eval_ep)
        print('logging episodic return evaluation', self.total_steps)
        # self.writer.add_scalar('episodic_return_eval', np.mean(episodic_returns), self.total_steps)