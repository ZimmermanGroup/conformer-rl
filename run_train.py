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

from alkanes import *

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env

import envs
from models import RTGN

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class A2CEvalAgent(A2CAgent):
    def eval_step(self, state):
        prediction = self.network(self.config.state_normalizer(state))
        return prediction['a']

class A2CRecurrentEvalAgent(A2CRecurrentAgent):
    def eval_step(self, state, done, rstates):
        with torch.no_grad():
            if done:
                prediction, rstates = self.network(self.config.state_normalizer(state))
            else:
                prediction, rstates = self.network(self.config.state_normalizer(state), rstates)

            out = to_np(prediction['a'])
            return out, rstates

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        done = True
        rstates = None
        while True:
            action, rstates = self.eval_step(state, done, rstates)

            done = False
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

class AdaTask:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):

        print ("seed is ", seed)
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

class DummyNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)

    def __call__(self, x):
        return x


def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 1
    config.task_fn = lambda: AdaTask('AllEightTorsionSetDense-v0', seed=random.randint(0,7e4))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=7e-5, alpha=0.99, eps=1e-5) #learning_rate #alpha #epsilon
    config.network = model
    config.discount = 0.9999 # gamma
    config.use_gae = False
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.0005 #ent_coef
    config.rollout_length = 5 # n_steps
    config.gradient_clip = 0.5 #max_grad_norm
    config.max_steps = 1000000
    config.save_interval = 10000
    config.eval_interval = 2000
    config.eval_episodes = 2
    config.eval_env = AdaTask('DiffDense-v0', seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()

    agent = A2CRecurrentEvalAgent(config)
    return agent

model = RTGN(6, 64, edge_dim=2)
model.to(torch.device('cuda'))


mkdir('log')
mkdir('tf_log')
set_one_thread()
select_device(0)
tag='eight_torsion_dense'
print(tag)
agent = a2c_feature(tag=tag)

run_steps(agent)
