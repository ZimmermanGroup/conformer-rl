from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

import logging
import numpy as np

from .graph_components import MPNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RTGNRecurrent(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, edge_dim, node_dim):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = _RTGNActorRecurrent(action_dim, hidden_dim, edge_dim=edge_dim, node_dim=node_dim)
        self.critic = _RTGNCriticRecurrent(action_dim, hidden_dim, edge_dim=edge_dim, node_dim=node_dim)

    def forward(self, obs, states=None, action=None):
        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list += b.to_data_list()
            nr_list.append(torch.LongTensor(nr))

        data = Batch.from_data_list(data_list)
        data = data.to(device)
        N = data.num_graphs

        so_far = 0
        torsion_batch_idx = []
        torsion_list_sizes = []

        for i in range(N):
            nr_list[i] += so_far
            so_far += int((data.batch == i).sum())
            torsion_batch_idx.extend([i]*int(nr_list[i].shape[0]))
            torsion_list_sizes += [nr_list[i].shape[0]]

        nrs = torch.cat(nr_list).to(device)
        torsion_batch_idx = torch.LongTensor(torsion_batch_idx).to(device)
        obs = (data, nrs, torsion_batch_idx, torsion_list_sizes)

        if states:
            hp, cp, hv, cv = states
            policy_states = (hp, cp)
            value_states = (hv, cv)
        else:
            policy_states = None
            value_states = None

        logits, (hp, cp) = self.actor(obs, policy_states)
        v, (hv, cv) = self.critic(obs, value_states)
        v = v.squeeze(0)

        dist = torch.distributions.Categorical(logits=logits)
        if action == None:
            action = dist.sample()

        tls_max = np.array(torsion_list_sizes).max()
        log_prob = dist.log_prob(action[:,:tls_max])

        entropy = dist.entropy()

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, (hp, cp, hv, cv)

class _RTGNCriticRecurrent(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, edge_dim, node_dim):
        super().__init__()
        self.mpnn = MPNN(hidden_dim=hidden_dim, edge_dim=edge_dim, node_dim=node_dim)
        self.set2set = gnn.Set2Set(hidden_dim, processing_steps=6)

        self.memory = nn.LSTM(2*hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.hidden_dim = hidden_dim

    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        if states:
            hx, cx = states
        else:
            hx = torch.zeros(1, N, self.hidden_dim).to(device)
            cx = torch.zeros(1, N, self.hidden_dim).to(device)

        out = self.mpnn(data)
        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1, N, 2*self.hidden_dim), (hx, cx))
        v = self.mlp(lstm_out)

        return v, (hx, cx)

class _RTGNActorRecurrent(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, edge_dim, node_dim):
        super().__init__()
        self.mpnn = MPNN(hidden_dim=hidden_dim, edge_dim=edge_dim, node_dim=node_dim)
        self.set2set = gnn.Set2Set(hidden_dim, processing_steps=6)

        self.memory = nn.LSTM(2*hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(5*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

        self.hidden_dim = hidden_dim

    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        if states:
            hx, cx = states
        else:
            hx = torch.zeros(1, N, self.hidden_dim).to(device)
            cx = torch.zeros(1, N, self.hidden_dim).to(device)

        out = self.mpnn(data)
        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.num_graphs,-1), (hx, cx))

        lstm_out = torch.index_select(
            lstm_out,
            dim=1,
            index=nrbidx
        )

        lstm_out = lstm_out.view(-1, self.hidden_dim)

        out = torch.index_select(
            out,
            dim=0,
            index=nonring.view(-1)
        )
        out = out.view(-1, self.hidden_dim * 4)
        out = torch.cat([lstm_out,out],1)   # shape = (num_torsions, 5*self.hidden_dim)
        out = self.mlp(out)

        logit = out.split(torsion_list_sizes)
        logit = torch.nn.utils.rnn.pad_sequence(logit).permute(1, 0, 2)

        return logit, (hx, cx)
