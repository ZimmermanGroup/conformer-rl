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

class TorsionGraphNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(TorsionGraphNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(4 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.lin3 = torch.nn.Linear(2* dim, 1)


    def forward(self, obs):
        obs = obs[0]
        data, nonring = obs
        data.to(torch.device(0))
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        out = torch.index_select(out, dim=0, index=torch.LongTensor(nonring).cuda().view(-1))
        out = out.view(4*out.shape[1],-1).permute(1, 0)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        v = self.lin3(pool)

        dist = torch.distributions.Categorical(logits=out)
        action = dist.sample().cuda()
        log_prob = dist.log_prob(action).unsqueeze(0).cuda()
        entropy = dist.entropy().unsqueeze(0).cuda()


        return {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

class RecurrentTorsionGraphNetv2(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(RecurrentTorsionGraphNetv2, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.lin3 = torch.nn.Linear(3* dim, 1)

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

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,1,-1), (hx, cx))

        out = torch.index_select(out, dim=0, index=torch.LongTensor(nonring).cuda().view(-1))
        out = out.view(4*out.shape[1],-1)
        out = out.permute(1, 0)
        out = torch.cat([out, torch.repeat_interleave(lstm_out, out.shape[0]).view(out.shape[0],-1)], dim=1)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        catted = torch.cat((pool, lstm_out.view(1,-1)), dim=1)
        v = self.lin3(catted)

        dist = torch.distributions.Categorical(logits=out)
        action = dist.sample().cuda()
        log_prob = dist.log_prob(action).unsqueeze(0).cuda()
        entropy = dist.entropy().unsqueeze(0).cuda()

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, (hx, cx)