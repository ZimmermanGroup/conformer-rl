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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from concurrent.futures import ProcessPoolExecutor


from models import RTGNBatch
from deep_rl import *
import envs

model = RTGNBatch(6, 128, edge_dim=1)
model.load_state_dict(torch.load('data/A2CRecurrentEvalAgent-batch_three_set-1000000.model'))
model.to(torch.device('cuda'))


def loaded_policy(env):
    env = gym.make(env)
    b, nr = env.reset()
    total_reward = 0
    start = True
    done = False
    step = 0
    while not done:
        state = [(b, nr)]

        with torch.no_grad():
            if start:
                print(state)
                prediction, rstates = model(state)
                start = False
            else:
                prediction, rstates = model(state, rstates)

        choice = prediction['a']
        step += 1
        print(step)
        state, rew, done, _ = env.step(to_np(choice))
        total_reward += rew

    return total_reward


loaded_policy('TrihexylEval-v0')
