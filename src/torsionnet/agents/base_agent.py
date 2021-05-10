import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torsionnet.utils import current_time, save_model, mkdir
from torsionnet.logging import TrainLogger, EnvLogger

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.train_env
        self.unique_tag = f'{config.tag}_{current_time()}'

        self.eval_logger = EnvLogger(tag=self.unique_tag)
        self.train_logger = TrainLogger(tag=self.unique_tag, use_cache=False, use_print=False)

        self.total_steps = 0
        


    def run_steps(self):
        config = self.config

        while self.total_steps < config.max_steps:
            if (config.save_interval > 0) and (self.total_steps % config.save_interval == 0):
                path = 'data/' + 'models/' + self.unique_tag + '/'
                mkdir(path)
                save_model(self.network, path + str(self.total_steps) + ".model")
            if (config.eval_interval > 0) and (self.total_steps % config.eval_interval == 0):
                self.eval_episodes()
            self.step()

        self.task.close()

    def step(self):
        raise NotImplementedError

    def eval_episode(self):
        raise NotImplementedError

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            self.eval_ep = ep
            total_rewards = self.eval_episode()
            episodic_returns.append(total_rewards)
            self.eval_logger.log_episode({"total_rewards": total_rewards})
            subdir = f'agent_step_{self.total_steps}/ep_{self.eval_ep}/'
            self.eval_logger.save_episode(subdir, save_molecules=True)
        print('logging episodic return evaluation', self.total_steps)
        self.train_logger.add_scalar('episodic_return_eval', np.mean(episodic_returns), self.total_steps)