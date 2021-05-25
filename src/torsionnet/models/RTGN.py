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

class RTGN(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=7, point_dim=3):
        super().__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorBatchNet(action_dim, dim, edge_dim=edge_dim, num_features=num_features)
        self.critic = CriticBatchNet(action_dim, dim, edge_dim=edge_dim, num_features=num_features)

    def forward(self, obs, action=None):
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

        logits = self.actor(obs)
        v = self.critic(obs)

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

        return prediction

class CriticBatchNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=1, num_features=3):
        super().__init__()
        self.mpnn = MPNN(edge_dim, dim, num_features)
        self.set2set = gnn.Set2Set(dim, processing_steps=6)

        self.mlp = nn.Sequential(nn.Linear(2*dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

        self.dim = dim

    def forward(self, obs):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        out = self.mpnn(data)
        pool = self.set2set(out, data.batch)
        v = self.mlp(pool)

        return v

class ActorBatchNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=1, num_features=3):
        super().__init__()
        self.mpnn = MPNN(edge_dim, dim, num_features)
        self.set2set = gnn.Set2Set(dim, processing_steps=6)

        self.fc = nn.Linear(2*dim, dim)
        self.mlp = nn.Sequential(nn.Linear(5*dim, dim), nn.ReLU(), nn.Linear(dim, action_dim))

        self.dim = dim

    def forward(self, obs):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        out = self.mpnn(data)
        pool = self.set2set(out, data.batch)
        graph_embed = self.fc(pool)

        graph_embed = torch.index_select(
            graph_embed,
            dim=0,
            index=nrbidx
        )

        graph_embed = graph_embed.view(-1, self.dim)

        out = torch.index_select(
            out,
            dim=0,
            index=nonring.view(-1)
        )
        out = out.view(-1, self.dim * 4)
        out = torch.cat([graph_embed,out],1)   # shape = (num_torsions, 5*self.dim)
        out = self.mlp(out)

        logit = out.split(torsion_list_sizes)
        logit = torch.nn.utils.rnn.pad_sequence(logit).permute(1, 0, 2)

        return logit
