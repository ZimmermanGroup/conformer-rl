import numpy as np
import random
import argparse
import torch

from main.utils import *
from main.agents import PPORecurrentAgent
from main.config import Config
from main.environments import Task
from main.models import RTGNBatch

from generate_molecule import DIFF, XORGATE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ppo_feature(tag, model):
    config = Config()
    config.tag = tag

    # Global Settings
    config.network = model
    config.hidden_size = model.dim

    # Batch Hyperparameters
    config.num_workers = 20
    config.rollout_length = 20
    config.recurrence = 2
    config.optimization_epochs = 4
    config.max_steps = 10000000
    config.save_interval = config.num_workers*200*5
    config.eval_interval = config.num_workers*200*5
    config.eval_episodes = 2
    config.mini_batch_size = 40

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # Task Settings
    config.train_env = Task('ConfEnv-v1', concurrency=True, num_envs=config.num_workers, seed=random.randint(0,1e5), mol_config=XORGATE, max_steps=300)
    config.eval_env = Task('ConfEnv-v1', seed=random.randint(0,7e4), mol_config=XORGATE, max_steps=300)
    config.curriculum = None

    return PPORecurrentAgent(config)


if __name__ == '__main__':
    nnet = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    nnet.to(device)
    set_one_thread()
    tag = 'xorgate-2'
    agent = ppo_feature(tag=tag, model=nnet)
    agent.run_steps()
