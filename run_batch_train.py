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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 30
    single_process = (config.num_workers == 1)
    config.task_fn = lambda: AdaTask('LigninTwoSet-v0', num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
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
    config.save_interval = 10000
    # config.eval_interval = 5000
    # config.eval_episodes = 1
    config.eval_env = AdaTask('LigninThreeSet-v0', seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()

    agent = A2CRecurrentEvalAgent(config)
    return agent

if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=1)
    model.load_state_dict(torch.load('data/A2CRecurrentEvalAgent-batch_three_set-1000000.model'))

    model.to(torch.device('cuda'))
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    select_device(0)
    tag='lignin_two_preloaded_gibbs'
    agent = a2c_feature(tag=tag)
    print(tag)
    run_steps(agent)