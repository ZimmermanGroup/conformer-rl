from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

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


class RTGNActorNet(torch.nn.Module):
    def __init__(self, action_dim, dim, edge_dim):
        super(RTGNActorNet, self).__init__()
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
    def __init__(self, action_dim, dim, edge_dim):
        super(RTGNCriticNet, self).__init__()
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
    def __init__(self, action_dim, dim, edge_dim=7):
        super(RTGN, self).__init__()
        num_features = 3
        self.action_dim = action_dim
        self.dim = dim

        self.actor = RTGNActorNet(action_dim, dim, edge_dim=edge_dim)
        self.critic = RTGNCriticNet(action_dim, dim, edge_dim=edge_dim)

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


class ActorNetAblation(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(ActorNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(6 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, action_dim)

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

        out = torch.index_select(out, dim=0, index=nonring.view(-1))
        out = out.view(4*out.shape[1],-1)
        out = out.permute(1, 0)
        out = torch.cat([out, torch.repeat_interleave(pool, out.shape[0]).view(out.shape[0],-1)], dim=1)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out, (None, None)

class CriticNetAblation(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(CriticNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, dim), nn.ReLU(), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(2*dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)

        self.action_dim = action_dim
        self.dim = dim

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
        out = F.relu(self.lin1(pool.view(1,-1)))
        v = self.lin3(out)

        return v, (None, None)


class RTGNAblation(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(RTGNAblation, self).__init__()
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
        data.to(torch.device(0))

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()
            cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()

        out = F.relu(self.lin0(data.x.cuda()))
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
        data.to(torch.device(0))

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()
            cx = Variable(torch.zeros(1, data.num_graphs, self.dim)).cuda()

        out = F.relu(self.lin0(data.x.cuda()))
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
    def __init__(self, action_dim, dim, edge_dim=7):
        super(RTGNBatch, self).__init__()
        num_features = 3
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorBatchNet(action_dim, dim, edge_dim=edge_dim)
        self.critic = CriticBatchNet(action_dim, dim, edge_dim=edge_dim)

    def forward(self, obs, states=None):
        data_list = []
        nr_list = []
        for b, nr in obs:
            data_list += b.to_data_list()
            nr_list.append(torch.LongTensor(nr).cuda())

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
        torsion_batch_idx = torch.LongTensor(torsion_batch_idx).cuda()
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


class ActorTorsionNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(ActorTorsionNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.memory = nn.LSTM(6 * dim, 6 * dim)

        self.lin1 = torch.nn.Linear(6 * dim, 2 * dim)
        self.lin2 = torch.nn.Linear(2 * dim, action_dim)


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
            hx = Variable(torch.zeros(1, nonring.shape[0], 6 * self.dim)).cuda()
            cx = Variable(torch.zeros(1, nonring.shape[0], 6 * self.dim)).cuda()

        out = F.relu(self.lin0(data.x.cuda()))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)

        out = torch.index_select(out, dim=0, index=nonring.view(-1))
        out = out.view(4*out.shape[1],-1)
        out = out.permute(1, 0)
        out = torch.cat([out, torch.repeat_interleave(pool, out.shape[0]).view(out.shape[0],-1)], dim=1)
        out, (hx, cx) = self.memory(out.unsqueeze(0))
        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out, (hx, cx)

class CriticTorsionNet(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(CriticTorsionNet, self).__init__()
        num_features = 3
        self.lin0 = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = gnn.Set2Set(dim, processing_steps=6)
        self.lin1 = torch.nn.Linear(2*dim, dim)
        self.lin3 = torch.nn.Linear(dim, 1)

        self.action_dim = action_dim
        self.dim = dim

        self.memory = nn.LSTM(2*dim, 2*dim)

    def forward(self, obs, states=None):
        obs = obs[0]
        data, nonring = obs
        data.to(torch.device(0))

        if states:
            hx, cx = states
        else:
            hx = Variable(torch.zeros(1, 1, 2 * self.dim)).cuda()
            cx = Variable(torch.zeros(1, 1, 2 * self.dim)).cuda()

        out = F.relu(self.lin0(data.x.cuda()))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        pool = self.set2set(out, data.batch)
        pool, (hx, cx) = self.memory(pool.unsqueeze(0))
        out = F.relu(self.lin1(pool.squeeze(0)))
        v = self.lin3(out)

        return v, (hx, cx)

class RTGNTorsionMemory(torch.nn.Module):
    def __init__(self, action_dim, dim):
        super(RTGNTorsionMemory, self).__init__()
        num_features = 3
        self.action_dim = action_dim
        self.dim = dim

        self.actor = ActorTorsionNet(action_dim, dim)
        self.critic = CriticTorsionNet(action_dim, dim)

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
