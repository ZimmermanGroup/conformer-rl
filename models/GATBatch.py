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

class ActorGAT(torch.nn.Module):
    def __init__(self, action_dim, dim, num_layers=6, num_features=3):
        super(ActorGAT, self).__init__()

        self.lin0 = torch.nn.Linear(num_features, dim)

        layers = nn.ModuleList()
        layers.append(gnn.GATConv(dim, dim, heads=2))

        for i in range(num_layers - 2):
            layers.append(gnn.GATConv(dim * 2, dim, heads=2))

        layers.append(gnn.GATConv(dim * 2, dim))
        self.conv_layers = layers

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.memory = nn.LSTM(2*dim, dim)

        self.action_dim = action_dim
        self.dim = dim


    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs
        
        if torch.cuda.is_available():
            data.to(torch.device(0))
            data.x = data.x.cuda()

        out = F.relu(self.lin0(data.x))

        if states:
            hx, cx = states
        else:
            if torch.cuda.is_available():
                hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()
                cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()
            else:
                hx = Variable(torch.zeros(1, data.num_graphs, self.dim))
                cx = Variable(torch.zeros(1, data.num_graphs, self.dim))    

        for layer in self.conv_layers:
            out = layer(out, data.edge_index)
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

class CriticGAT(torch.nn.Module):
    def __init__(self, action_dim, dim, num_layers=6, num_features=3):
        super(CriticGAT, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        layers = nn.ModuleList()
        layers.append(gnn.GATConv(dim, dim, heads=2))

        for i in range(num_layers - 2):
            layers.append(gnn.GATConv(dim * 2, dim, heads=2))

        layers.append(gnn.GATConv(dim * 2, dim))
        self.conv_layers = layers

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)

        self.action_dim = action_dim
        self.dim = dim

        self.memory = nn.LSTM(2*dim, dim)

    def forward(self, obs, states=None):
        data, nonring, nrbidx, torsion_list_sizes = obs

        if torch.cuda.is_available():
            data.to(torch.device(0))
            data.x = data.x.cuda()

        out = F.relu(self.lin0(data.x))

        if states:
            hx, cx = states
        else:
            if torch.cuda.is_available():
                hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()
                cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()
            else:
                hx = Variable(torch.zeros(1, data.num_graphs, self.dim))
                cx = Variable(torch.zeros(1, data.num_graphs, self.dim))               

        for layer in self.conv_layers:
            out = layer(out, data.edge_index)

        pool = self.set2set(out, data.batch)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.num_graphs,-1), (hx, cx))
        out = F.relu(self.lin1(lstm_out))
        v = self.lin3(out)

        return v, (hx, cx)

class GATBatch(torch.nn.Module):
    def __init__(self, action_dim, dim, point_dim=3, num_layers=6):
        super(GATBatch, self).__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorGAT(action_dim, dim, num_features=num_features, num_layers=num_layers)
        self.critic = CriticGAT(action_dim, dim, num_features=num_features, num_layers=num_layers)

    def forward(self, obs, states=None, action=None):

        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list += b.to_data_list()

            if torch.cuda.is_available():
                nr_list.append(torch.LongTensor(nr).cuda())
            else:
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

        nrs = torch.cat(nr_list)

        if torch.cuda.is_available():
            torsion_batch_idx = torch.LongTensor(torsion_batch_idx).cuda()
        else:
            torsion_batch_idx = torch.LongTensor(torsion_batch_idx)

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
        if action == None:
            action = dist.sample()

        try:
            tls_max = np.array(torsion_list_sizes).max()
            log_prob = dist.log_prob(action[:,:tls_max]).unsqueeze(0)
        except RuntimeError as e:
            logging.error(e)
            logging.error(f'action shape {action.shape}')
            logging.error(f'logits shape {dist.logits.shape}')
            raise Exception()


        if torch.cuda.is_available():
            entropy = dist.entropy().unsqueeze(0).cuda()
        else:
            entropy = dist.entropy().unsqueeze(0)

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, (hp, cp, hv, cv)