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
import envs

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


#message passing
from torch.autograd import Variable
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F



class CriticTransformer(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(CriticTransformer, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)

        self.action_dim = action_dim
        self.dim = dim

        self.memory = nn.LSTM(2*dim, dim)

    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.shape[0], self.dim)).cuda()
            cx = Variable(torch.zeros(1, data.shape[0], self.dim)).cuda()

        out = F.relu(self.lin0(data))
        out = self.transformer_encoder(out)

        num_atoms = out.shape[1]
        batch_len = out.shape[0]
        bidxs = []

        for i in range(batch_len):
            bidxs.append(torch.ones([num_atoms], dtype=torch.int32) * i)

        bidxs = torch.cat(bidxs).long().cuda()
        out = torch.flatten(out, start_dim=0, end_dim=1)

        pool = self.set2set(out, bidxs)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.shape[0],-1), (hx, cx))

        out = F.relu(self.lin1(lstm_out))
        v = self.lin3(out)

        return v, (hx, cx)

class ActorTransformer(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(ActorTransformer, self).__init__()
        num_features = 3

        self.lin0 = torch.nn.Linear(num_features, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.memory = nn.LSTM(2*dim, dim)

        self.action_dim = action_dim
        self.dim = dim


    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.shape[0], self.dim)).cuda()
            cx = Variable(torch.zeros(1, data.shape[0], self.dim)).cuda()

        out = F.relu(self.lin0(data))
        out = self.transformer_encoder(out)

        num_atoms = out.shape[1]
        batch_len = out.shape[0]
        bidxs = []

        for i in range(batch_len):
            bidxs.append(torch.ones([num_atoms], dtype=torch.int32) * i)

        bidxs = torch.cat(bidxs).long().cuda()
        out = torch.flatten(out, start_dim=0, end_dim=1)

        pool = self.set2set(out, bidxs)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.shape[0],-1), (hx, cx))

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

class GraphTransformerBatch(torch.nn.Module):
    def __init__(self, action_dim, dim, point_dim=3):
        super(GraphTransformerBatch, self).__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorTransformer(action_dim, dim)
        self.critic = CriticTransformer(action_dim, dim)

    def forward(self, obs, states=None, action=None):
        # data_list = []
        # nr_list = []
        # for b, nr in obs:
        #     data_list.append(b.x.cuda())
        #     nr_list.append(torch.LongTensor(nr).cuda())

        # torsion_batch_idx = []
        # torsion_list_sizes = []

        # for i in range(len(obs)):
        #     trues = (b.batch == i).view(1, -1)
        #     nr_list[i] += so_far
        #     so_far += int((b.batch == i).sum())
        #     torsion_batch_idx.extend([i]*int(nr_list[i].shape[0]))
        #     torsion_list_sizes += [nr_list[i].shape[0]]

        # nrs = torch.cat(nr_list)
        # torsion_batch_idx = torch.LongTensor(torsion_batch_idx).cuda()
        # obs = (b, nrs, torsion_batch_idx, torsion_list_sizes)

        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list.append(b.x.cuda())
            nr_list.append(torch.LongTensor(nr).cuda())

        b = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True)
        nrs = torch.nn.utils.rnn.pad_sequence(nr_list, batch_first=True)

#         print(b.shape) #torch.Size([4, 15, 3])
        max_atoms = b.shape[1]

#         print(nrs.shape) #4,10,4 batch first

        torsion_batch_idx = []
        torsion_list_sizes = []

        for i in range(len(obs)):
            nrs[i] += max_atoms
            torsion_batch_idx.extend([i]*int(nr_list[i].shape[0]))
            torsion_list_sizes += [nr_list[i].shape[0]]

        torsion_batch_idx = torch.LongTensor(torsion_batch_idx).cuda()
        obs = (b, nrs, torsion_batch_idx, torsion_list_sizes)

        if states:
            hp, cp, hv, cv = states
            policy_states = (hp, cp)
            value_states = (hv, cv)
        else:
            policy_states = None
            value_states = None

        logits, (hp, cp) = self.actor(obs, states=policy_states)
        v, (hv, cv) = self.critic(obs, states=value_states)

        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        action = dist.sample().cuda()
        log_prob = dist.log_prob(action).unsqueeze(0).cuda()
        entropy = dist.entropy().unsqueeze(0).cuda()

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, (hp, cp, hv, cv)


if __name__ == '__main__':
    from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
    import envs
    m = GraphTransformerBatch(6, 128)
    m.to(torch.device('cuda'))

    task = AdaTask('Diff-v0', seed=random.randint(0,7e4), num_envs=3, single_process=True)
    x = task.reset()
    prediction, recurrent_states = m(x)
    print(prediction)
    next_states, rewards, terminals, info = task.step(to_np(prediction['a']))
