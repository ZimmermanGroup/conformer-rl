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
from torch.autograd import Variable
from torch import nn

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env
from deep_rl.agent.PPO_recurrent_agent_gnn_recurrence import PPORecurrentAgentGnnRecurrence
from deep_rl.agent.PPO_recurrent_agent_gnn import PPORecurrentAgentGnn

import envs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

HIDDEN_SIZE = 128

class PPORecurrentEvalAgent(PPORecurrentAgentGnnRecurrence):
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
        print("environment resetting")

        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)
    
class DummyNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)

    def __call__(self, x):
        return x


class CriticBatchNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim):
        super(CriticBatchNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)

        self.action_dim = action_dim
        self.dim = dim

        self.memory = nn.LSTM(2*dim, dim)

    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs
        data.to(device)

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)
            cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)

        out = F.relu(self.lin0(data.x.to(device)))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.num_graphs,-1), (hx, cx))
        out = F.relu(self.lin1(lstm_out))
        v = self.lin3(out)

        return v, (hx, cx)

class ActorBatchNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim):
        super(ActorBatchNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.memory = nn.LSTM(2*dim, dim)

        self.action_dim = action_dim
        self.dim = dim

    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs
        data.to(device)

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)
            cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)

        out = F.relu(self.lin0(data.x.to(device)))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.num_graphs,-1), (hx, cx))

        lstm_out = torch.index_select(
            lstm_out,
            dim=1,
            index=nrbidx
        )
        out = torch.index_select(
            out,
            dim=0,
            index=nonring.view(-1)
        ).view(4, -1, self.dim)


        out = torch.cat([lstm_out,out],0)   #5, num_torsions, self.dim
        out = out.permute(2,1,0).reshape(-1, 5*self.dim) #num_torsions, 5*self.dim
        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        logit = out.split(torsion_list_sizes)
        logit = torch.nn.utils.rnn.pad_sequence(logit).permute(1,0,2)

        return logit, (hx, cx)

class RTGNBatch(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=1, point_dim=3):
        super(RTGNBatch, self).__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorBatchNet(action_dim, dim, edge_dim=edge_dim)
        self.critic = CriticBatchNet(action_dim, dim, edge_dim=edge_dim)

    def forward(self, obs, states=None, action=None):
        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list += b.to_data_list()
            nr_list.append(torch.LongTensor(nr).to(device))

        b = Batch.from_data_list(data_list)
        so_far = 0
        torsion_batch_idx = []
        torsion_list_sizes = []

        for i in range(b.num_graphs):
            trues = (b.batch == i).view(1, -1)
            nr_list[i] += so_far
            so_far += int((b.batch == i).sum())
            torsion_batch_idx.extend([i]*int(nr_list[i].shape[0]))
            torsion_list_sizes += [nr_list[i].shape[0]]

        nrs = torch.cat(nr_list)
        torsion_batch_idx = torch.LongTensor(torsion_batch_idx).to(device)
        obs = (b, nrs, torsion_batch_idx, torsion_list_sizes)

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
        if action is None:
            action = dist.sample()
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

model = RTGNBatch(6, HIDDEN_SIZE)
model.to(device)

def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    #Constant
    config.network = model
    config.hidden_size = HIDDEN_SIZE
    config.state_normalizer = DummyNormalizer()
    
    #Task
    config.task_fn = lambda: AdaTask('Diff-v0', num_envs = config.num_workers, single_process = False, seed=random.randint(0,7e4))
    config.eval_env = AdaTask('Diff-v0', seed=random.randint(0,7e4))

    #Batch
    config.num_workers = 20
    config.rollout_length = 200 # n_steps
    config.optimization_epochs = 10
    config.mini_batch_size = 20*20
    config.max_steps = 10000000
    config.save_interval = 10000
    config.eval_interval = 0
    config.eval_episodes = 2
    config.recurrence = 5

    #Coefficients
    lr = 7e-5 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5) #learning_rate #alpha #epsilon
    config.discount = 0.999 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0 #ent_coef
    config.gradient_clip = 0.5 #max_grad_norm
    config.ppo_ratio_clip = 0.2


    
    agent = PPORecurrentEvalAgent(config)
    return agent




mkdir('log')
mkdir('tf_log')
set_one_thread()
tag = "Diff-May7-V1"
agent = ppo_feature(tag=tag)

run_steps(agent)