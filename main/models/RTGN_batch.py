from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

import logging
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CriticBatchNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=1, num_features=3):
        super(CriticBatchNet, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(inplace=False), nn.Linear(dim, dim * dim))
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
        data.x = data.x.to(device)

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)
            cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)

        out = F.relu(self.lin0(data.x))
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
    def __init__(self, action_dim, dim, edge_dim=1, num_features=3):
        super(ActorBatchNet, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(inplace=False), nn.Linear(dim, dim * dim))
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
        data.x = data.x.to(device)

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)
            cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).to(device)

        out = F.relu(self.lin0(data.x))
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

        lstm_out = lstm_out.view(-1, self.dim)

        out = torch.index_select(
            out,
            dim=0,
            index=nonring.view(-1)
        )

        out = out.view(-1, self.dim * 4)
        out = torch.cat([lstm_out,out],1)   #5, num_torsions, self.dim

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        logit = out.split(torsion_list_sizes)
        logit = torch.nn.utils.rnn.pad_sequence(logit).permute(1, 0, 2)

        return logit, (hx, cx)

class RTGNBatch(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim=7, point_dim=3):
        super(RTGNBatch, self).__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorBatchNet(action_dim, dim, edge_dim=edge_dim, num_features=num_features)
        self.critic = CriticBatchNet(action_dim, dim, edge_dim=edge_dim, num_features=num_features)

    def forward(self, obs, states=None, action=None):
        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list += b.to_data_list()
            nr_list.append(torch.LongTensor(nr))

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

        nrs = torch.cat(nr_list).to(device)
        torsion_batch_idx = torch.LongTensor(torsion_batch_idx).to(device)
        obs = (b, nrs, torsion_batch_idx, torsion_list_sizes)

        if states:
            hp, cp, hv, cv = states
            hp, cp, hv, cv = hp.to(device), cp.to(device), hv.to(device), cv.to(device)
            policy_states = (hp, cp)
            value_states = (hv, cv)
        else:
            policy_states = None
            value_states = None

        logits, (hp, cp) = self.actor(obs, policy_states)
        v, (hv, cv) = self.critic(obs, value_states)

        dist = torch.distributions.Categorical(logits=logits)
        if action == None:
            action = dist.sample()

        action = action.to(device)

        try:
            tls_max = np.array(torsion_list_sizes).max()
            log_prob = dist.log_prob(action[:,:tls_max]).unsqueeze(0)
        except RuntimeError as e:
            logging.error(e)
            logging.error(f'action shape {action.shape}')
            logging.error(f'logits shape {dist.logits.shape}')
            raise Exception()

        entropy = dist.entropy().unsqueeze(0)

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, (hp, cp, hv, cv)
