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
import time

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


from models import *
from deep_rl import *
import envs


# conformer-ml/data/A2CRecurrentEvalAgent-obabel_sets_seven_energy_sum_rewardnorm-50000.model
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
        print('step', step)
        state, rew, done, info = env.step(to_np(choice))
        total_reward += float(rew)
        print('rew', rew)
        print('total_reward', total_reward)

    if isinstance(info, tuple):
        for i, info_ in enumerate(info):
            print('episodic_return', info_['episodic_return'])
    else:
        print('episodic_return', info['episodic_return'])
    return total_reward


if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=1)
    # model = GATBatch(6, 128)
    # model = GraphTransformerBatch(6, 128)
    model.load_state_dict(torch.load('data/A2CRecurrentEvalAgent-rtgn_pruning_fix_3set-2520000.model'))
    model.to(torch.device('cuda'))

    outputs = []
    times = []
    for i in range(10):
        start = time.time()
        output = loaded_policy(model, 'TrihexylUnique-v0')
        end = time.time()
        outputs.append(output)
        times.append(end - start)
    print('outputs', outputs)
    print('times', times)
