from os import environ

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import logging
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

from utils import *

import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env

import envs
from models import *

from a2crecurrentziping import A2CRecurrentCurriculumAgent

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Curriculum():
    def __init__(self, win_cond=0.7, success_percent=0.7, fail_percent=0.2, min_length=100):
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

    config.num_workers = int(environ['SLURM_CPUS_PER_TASK'])
    single_process = (config.num_workers == 1)
    # single_process = True

    config.task_fn = lambda: AdaTask('TestSetCurriculaExternPoints-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)

    # config.task_fn = lambda: AdaTask('TestPruningSetGibbsEdit-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
    # config.task_fn = lambda: AdaTask('TestPruningSetCurriculaExtern-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
    # config.task_fn = lambda: AdaTask('AllTenTorsionSetPruning-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
    config.linear_lr_scale = False
    if config.linear_lr_scale:
        lr = 7e-5 * config.num_workers
    else:
        lr = 7e-5 * np.sqrt(config.num_workers)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.network = model
    config.hidden_dim = model.dim
    config.discount = 0.9999 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.0001 #ent_coef
    config.rollout_length = 5 # n_steps
    config.gradient_clip = 0.5 #max_grad_norm
    config.max_steps = 10000000
    config.save_interval = config.num_workers * 200 * 5

    config.curriculum = Curriculum(min_length=config.num_workers)

    # config.eval_interval = config.num_workers * 200 * 5
    # config.eval_episodes = 1
    # config.eval_env = AdaTask('DiffUnique-v0', seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    # config.reward_normalizer = MeanStdNormalizer()

    agent = A2CRecurrentEvalAgent(config)
    return agent

def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = int(environ['SLURM_CPUS_PER_TASK'])
    single_process = (config.num_workers == 1)
    config.linear_lr_scale = False
    if config.linear_lr_scale:
        lr = 2e-5 * config.num_workers
    else:
        lr = 2e-5 * np.sqrt(config.num_workers)

    config.curriculum = Curriculum(min_length=config.num_workers)

    config.task_fn = lambda: AdaTask('LigninAllSetPruningSkeletonCurriculumLong-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process) # causes error

    # config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.network = model
    config.hidden_size = model.dim
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.state_normalizer = DummyNormalizer()
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    # config.mini_batch_size = config.rollout_length * config.num_workers
    config.mini_batch_size = 25
    config.ppo_ratio_clip = 0.2
    config.save_interval = config.num_workers * 1000 * 2
    config.eval_interval = config.num_workers * 1000 * 2
    config.eval_episodes = 1
    config.eval_env = AdaTask('LigninPruningSkeletonEvalFinalLong-v0', seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    run_steps(PPORecurrentEvalAgent(config))


if __name__ == '__main__':
    # model = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    # model = GraphTransformerBatch(6, 128, num_layers=12, point_dim=5)
    model = GATBatch(6, 128, num_layers=10, point_dim=5)
    # model.load_state_dict(torch.load('data/PPORecurrentEvalAgent-ppo_gat_pruning_lignin_log_curr_long-175000.model'))
    model.to(torch.device('cuda'))
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    select_device(0)
    tag = environ['SLURM_JOB_NAME']
    agent = ppo_feature(tag=tag)
    # agent = a2c_feature(tag=tag)
    logging.info(tag)
    run_steps(agent)
