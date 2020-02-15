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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

env_name = 'OneSet-v0'

class PPOEvalAgent(PPOAgent):
    def eval_step(self, state):
        prediction = self.network(self.config.state_normalizer(state))
        return prediction['a']

class PPORecurrentEvalAgent(PPORecurrentAgent):
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


from torch.autograd import Variable
from torch import nn

class ActorNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(ActorNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.memory = nn.LSTM(2*dim, dim)    
        
        self.action_dim = action_dim
        self.dim = dim
        

    def forward(self, obs, states=None):
        obs = obs[0]
        data, nonring = obs
        data.to(device)
        nonring = torch.LongTensor(nonring).to(device)
        
        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, self.dim)).to(device)
            cx = Variable(torch.zeros(1, 1, self.dim)).to(device)
    
        out = F.relu(self.lin0(data.x)).to(device)
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,1,-1), (hx, cx))        
        
        out = torch.index_select(out, dim=0, index=nonring.view(-1))
        out = out.view(4*out.shape[1],-1)
        out = out.permute(1, 0)
        out = torch.cat([out, torch.repeat_interleave(lstm_out, out.shape[0]).view(out.shape[0],-1)], dim=1)
#       
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        
        return out, (hx, cx)       
        
class CriticNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(CriticNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)
        
        self.action_dim = action_dim
        self.dim = dim
        
        self.memory = nn.LSTM(2*dim, dim)    

    def forward(self, obs, states=None):
        obs = obs[0]
        data, nonring = obs
        data.to(device)
        
        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, self.dim)).to(device)
            cx = Variable(torch.zeros(1, 1, self.dim)).to(device)
    
        out = F.relu(self.lin0(data.x)).to(device)
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,1,-1), (hx, cx))        
        
        out = F.relu(self.lin1(lstm_out.view(1,-1)))
        v = self.lin3(out)
        
        return v, (hx, cx)
        
        

class RTGN(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(RTGN, self).__init__()
        num_features = 3
        self.action_dim = action_dim
        self.dim = dim
        
        self.actor = ActorNet(action_dim, dim)
        self.critic = CriticNet(action_dim, dim)
        
    def forward(self, obs, states=None):
        
        if states:
            hp, cp, hv, cv = states

            policy_states = (hp, cp)
            value_states = (hv, cv)
        else:
            policy_states = None
            value_states = None
    
        logits, (hp, cp) = self.actor(obs, policy_states)
        v, (hv, cv) = self.critic(obs, value_states)
        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().to(device)
        log_prob = dist.log_prob(action).unsqueeze(0).to(device)
        entropy = dist.entropy().unsqueeze(0).to(device)

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }
        
        return prediction, (hp, cp, hv, cv)

model = RTGN(6, 128)
model.to(device)

def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 1
    config.task_fn = lambda: AdaTask(env_name, seed=random.randint(0,7e4))
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-5, alpha=0.99, eps=1e-5) #learning_rate #alpha #epsilon
    config.network = model
    config.discount = 0.9999 # gamma
    config.use_gae = False
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0 #ent_coef
    config.rollout_length = 5 # n_steps
    config.gradient_clip = 0.5 #max_grad_norm
    config.max_steps = 5000000
    config.save_interval = 10000
    config.eval_interval = 2000
    config.eval_episodes = 2
    config.eval_env = AdaTask(env_name, seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    config.ppo_ratio_clip = 0.9
    
    agent = PPORecurrentEvalAgent(config)
    return agent

mkdir('log')
mkdir('tf_log')
set_one_thread()
select_device(0)
tag='ppo:oneset'
agent = ppo_feature(tag=tag)

run_steps(agent)

