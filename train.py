import numpy as np
import random
import argparse
import torch

from main import mkdir
from main import PPORecurrentAgent
from main import Config
from main import Task
from main import RTGNBatch

from generate_molecule import DIFF, XORGATE

class Curriculum():
    def __init__(self, win_cond=0.7, success_percent=0.7, fail_percent=0.2, min_length=100):
        self.win_cond = win_cond
        self.success_percent = success_percent
        self.fail_percent = fail_percent
        self.min_length = min_length

    def return_win_cond():
        return self.win_cond

def ppo_feature(tag, model):
    config = Config()
    config.tag = tag
    

    config.num_workers = 4
    single_process = False
    lr = 5e-6 * np.sqrt(config.num_workers)

    config.curriculum = Curriculum(min_length=config.num_workers)

    config.train_env = Task('ConfEnv-v1', num_envs=config.num_workers, seed=random.randint(0,1e5), mol_config=XORGATE)

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.network = model
    config.hidden_size = model.dim
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.rollout_length = 2
    config.recurrence = 1
    config.optimization_epochs = 1
    # config.mini_batch_size = config.rollout_length * config.num_workers
    config.mini_batch_size = 25
    config.ppo_ratio_clip = 0.2
    config.save_interval = 3
    config.eval_interval = 3
    config.eval_episodes = 1
    config.eval_env = Task('ConfEnv-v1', seed=random.randint(0,7e4), mol_config=XORGATE)
    return PPORecurrentAgent(config)


if __name__ == '__main__':
    nnet = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    # model.to(torch.device('cuda'))
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    # set_one_thread()
    # select_device(0)
    tag = 'train_lignins'
    agent = ppo_feature(tag=tag, model=nnet)
    agent.run_steps()
