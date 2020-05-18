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

from pandas import DataFrame

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

def loaded_policy(model, env):
    num_envs = 1
    single_process = (num_envs == 1)

    env = AdaTask(env, seed=random.randint(0,7e4), num_envs=num_envs, single_process=single_process)
    state = env.reset()
    total_reward = 0
    start = True
    done = False
    step = 0
    while step < 200:
        with torch.no_grad():
            if start:
                prediction, rstates = model(state)
                start = False
            else:
                prediction, rstates = model(state, rstates)

        choice = prediction['a']
        step += 1
        state, rew, done, info = env.step(to_np(choice))
        total_reward += float(rew)

    if isinstance(info, tuple):
        for i, info_ in enumerate(info):
            print('episodic_return', info_['episodic_return'])
            episodic_return = info_['episodic_return']
    else:
        print('episodic_return', info['episodic_return'])
        episodic_return = info['episodic_return']
    return episodic_return


if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=1)
    outputs = []
    print('outputs', outputs)

    full_outputs = []
    for i in range(0, 10):
        model.load_state_dict(torch.load(f'transfer_test_t_chain/models/{i}.model'))
        model.to(torch.device('cuda'))

        for j in range(0, 10):
            samples = []

            for _ in range(3):
                samples.append(loaded_policy(model, f'TChainTest3-v{j}'))
            output = min(samples)
            outputs.append(output)
            print(i,j,output)

        full_outputs.append(outputs)
        outputs = []

    ans = np.array(full_outputs)

    df = DataFrame(ans)
    df.to_csv('energy_difference_t_chains_multisample.csv')
    print(df)
