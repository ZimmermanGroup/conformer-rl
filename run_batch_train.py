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

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env

import envs
from models import RTGN, RTGNBatch

from a2crecurrentziping import A2CRecurrentCurriculumAgent

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = int(environ['SLURM_CPUS_PER_TASK'])
    single_process = (config.num_workers == 1)
    config.task_fn = lambda: AdaTask('TChainTrain-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
    config.linear_lr_scale = False
    if config.linear_lr_scale:
        lr = 7e-5 * config.num_workers
    else:
        lr = 7e-5 * np.sqrt(config.num_workers)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.network = model
    config.discount = 0.9999 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.0001 #ent_coef
    config.rollout_length = 5 # n_steps
    config.gradient_clip = 0.5 #max_grad_norm
    config.max_steps = 5000000
    config.save_interval = config.num_workers * 200 * 5
    # config.eval_interval = config.num_workers * 200 * 5
    # config.eval_episodes = 1
    # config.eval_env = AdaTask('TChainTrain-v0', seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    # config.reward_normalizer = MeanStdNormalizer()

    agent = A2CRecurrentCurriculumAgent(config)
    return agent

if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=1)
    # model.load_state_dict(torch.load('data/A2CRecurrentEvalAgent-StraightChainTen-210000.model'))
    model.to(torch.device('cuda'))
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    select_device(0)
    tag = environ['SLURM_JOB_NAME']
    agent = a2c_feature(tag=tag)
    print(tag)
    run_steps(agent)
