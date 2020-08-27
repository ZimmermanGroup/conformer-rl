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

class CriticTransformer(torch.nn.Module):
    def __init__(self, action_dim, dim, num_layers=6, num_features=3):
        super(CriticTransformer, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        pool_in = torch.flatten(out, start_dim=0, end_dim=1)


        pool = self.set2set(pool_in, bidxs)
        lstm_out, (hx, cx) = self.memory(pool.view(1,data.shape[0],-1), (hx, cx))

        out = F.relu(self.lin1(lstm_out))
        v = self.lin3(out)

        return v, (hx, cx)

class ActorTransformer(torch.nn.Module):
    def __init__(self, action_dim, dim, num_layers=6, num_features=3):
        super(ActorTransformer, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(5 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

        self.memory = nn.LSTM(2*dim, dim)

        self.action_dim = action_dim
        self.dim = dim

    def default_mem(self, num_envs):
        hx = Variable(torch.zeros(1, num_envs, self.dim)).cuda()
        cx = Variable(torch.zeros(1, num_envs, self.dim)).cuda()

        return hx, cx

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

class GraphTransformerBatch(torch.nn.Module):
    def __init__(self, action_dim, dim, point_dim=3, num_layers=6):
        super(GraphTransformerBatch, self).__init__()
        num_features = point_dim
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorTransformer(action_dim, dim, num_layers=num_layers, num_features=num_features)
        self.critic = CriticTransformer(action_dim, dim, num_layers=num_layers, num_features=num_features)

    def forward(self, obs, states=None, action=None):
        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list.append(b.x.cuda())
            nr_list.append(torch.LongTensor(nr).cuda())

        b = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True)
        nrs = torch.nn.utils.rnn.pad_sequence(nr_list, batch_first=True)

        max_atoms = b.shape[1]

        torsion_batch_idx = []
        torsion_list_sizes = []

        for i in range(len(obs)):
            nrs[i] += max_atoms * i
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

        entropy = dist.entropy().unsqueeze(0).cuda()

        prediction = {
            'a': action,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }

        return prediction, (hp, cp, hv, cv)
