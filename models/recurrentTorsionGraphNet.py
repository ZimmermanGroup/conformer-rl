import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RTGNActorNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=1, num_features=3):
        super(RTGNActorNet, self).__init__()
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
        obs = obs[0]
        data, nonring = obs
        data.to(torch.device(0))
        nonring = torch.LongTensor(nonring).to(torch.device(0))

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, self.dim)).cuda()
            cx = Variable(torch.zeros(1, 1, self.dim)).cuda()

        out = F.relu(self.lin0(data.x.cuda()))
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

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out, (hx, cx)

class RTGNCriticNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=1, num_features=3):
        super(RTGNCriticNet, self).__init__()
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
        obs = obs[0]
        data, nonring = obs
        data.to(torch.device(0))

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, self.dim)).cuda()
            cx = Variable(torch.zeros(1, 1, self.dim)).cuda()

        out = F.relu(self.lin0(data.x.cuda()))
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
    def __init__(self, action_dim, dim, edge_dim=7, point_dim=3):
        super(RTGN, self).__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = RTGNActorNet(action_dim, dim, edge_dim=edge_dim, num_features=num_features)
        self.critic = RTGNCriticNet(action_dim, dim, edge_dim=edge_dim, num_features=num_features)

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