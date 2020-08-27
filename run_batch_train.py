from os import environ
import multiprocessing
import logging
import random
import gym

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

import deep_rl
from deep_rl.component.envs import DummyVecEnv, make_env
from deep_rl.utils.misc import mkdir, generate_tag, run_steps
from deep_rl.utils.torch_utils import select_device, set_one_thread
from deep_rl.utils.config import Config

from environment import graphenvironments
from environment import zipingenvs
from utils.agentUtilities import *
from models.recurrentTorsionGraphNetBatch import RTGNBatch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

gym.envs.register(
     id='MolTaskEnv-v0',
     entry_point='environment.graphenvironments:PruningSetGibbs',
     max_episode_steps=1000,
     kwargs = {"folder" : ENV_FOLDER}
)
gym.envs.register(
    id='MolEvalEnv-v0',
    entry_point='environment.graphenvironments:PruningSetGibbs',
    max_episode_steps=1000,
    kwargs= {"folder" : EVAL_FOLDER}
)

class Curriculum():
    def __init__(self, win_cond=0.01, success_percent=0.7, fail_percent=0.2, min_length=100):
        self.win_cond = win_cond
        self.success_percent = success_percent
        self.fail_percent = fail_percent
        self.min_length = min_length

    def return_win_cond():
        return self.win_cond

def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    #Global Settings
    config.network = model
    config.hidden_dim = model.dim
    config.state_normalizer = DummyNormalizer()

    #Task Settings
    config.task_fn = lambda: AdaTask('MolTaskEnv-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
    config.eval_env = AdaTask('MolEvalEnv-v0', seed=random.randint(0,7e4))
    config.curriculum = Curriculum(min_length=config.num_workers)

    #Batch Hyperparameters
    config.num_workers = 1#int(environ['SLURM_CPUS_PER_TASK'])
    single_process = (config.num_workers == 1)
    config.rollout_length = 5 # n_steps
    config.max_steps = 10000000
    config.save_interval = config.num_workers * 200 * 5
    config.eval_interval = config.num_workers * 200 * 5
    config.eval_episodes = 1

    #Coefficient Hyperparameters
    config.linear_lr_scale = False
    lr = 7e-5 * config.num_workers if config.linear_lr_scale else 7e-5 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.discount = 0.9999 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.0001 #ent_coef
    config.gradient_clip = 0.5 #max_grad_norm
    # config.reward_normalizer = MeanStdNormalizer()

    agent = A2CRecurrentEvalAgent(config)
    return agent

def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    #Global Settings
    config.network = model
    config.hidden_size = model.dim
    config.state_normalizer = DummyNormalizer()
    
    #Task Settings
    config.task_fn = lambda: AdaTask('MolTaskEnv-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process) # causes error
    config.eval_env = AdaTask('MolEvalEnv-v0', seed=random.randint(0,7e4))

    #Batch Hyperparameters
    config.num_workers = 1#int(environ['SLURM_CPUS_PER_TASK'])
    single_process = (config.num_workers == 1)
    config.save_interval = config.num_workers * 200 * 5
    config.eval_interval = config.num_workers * 200 * 5
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.eval_episodes = 1
    # config.mini_batch_size = config.rollout_length * config.num_workers
    config.mini_batch_size = 25

    #Coefficient Hyperparameters
    config.linear_lr_scale = False
    lr = 7e-5 * config.num_workers if config.linear_lr_scale else 7e-5 * np.sqrt(config.num_workers)
    config.curriculum = Curriculum(min_length=config.num_workers)
    # config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    run_steps(PPORecurrentEvalAgent(config))


if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    ENV_FOLDER = "molecules/trihexyl/"
    EVAL_FOLDER = "molecules/diff/"
    # model = GraphTransformerBatch(6, 128, num_layers=12)
    # model = GATBatch(6, 128, num_layers=10, point_dim=5)
    # model.load_state_dict(torch.load('data/A2CRecurrentEvalAgent-StraightChainTen-210000.model'))
    model.to(device)
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    # select_device(0)
    # tag = environ['SLURM_JOB_NAME']
    tag = "test";
    agent = ppo_feature(tag=tag)
    logging.info(tag)
    run_steps(agent)
