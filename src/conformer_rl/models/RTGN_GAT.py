"""
RTGN_GAT
========
"""
from torch import nn
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

import numpy as np

from typing import List, Tuple, Dict

from conformer_rl.models.graph_components import GAT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RTGNGat(torch.nn.Module):
    """Actor-critic neural network using graph transformer network (GAT) [1]_
    for predicting discrete torsion angles.

    Works with molecules with any number of torsion angles, and batches containing graphs of different molecules.

    Parameters
    ----------
    action_dim : int
        The number of discrete action choices for each torsion angle.
    hidden_dim : int
        Dimension of the hidden layer.
    node_dim : int
        The dimension of each node embedding in the input graph.

    References
    ----------
    .. [1] `GAT paper <https://arxiv.org/abs/1911.06455>`_
    """
    def __init__(self, action_dim: int, hidden_dim: int, node_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = _RTGNGatActor(action_dim, hidden_dim, node_dim=node_dim)
        self.critic = _RTGNGatCritic(action_dim, hidden_dim, node_dim=node_dim)

    def forward(self, 
    obs: List[Tuple[Batch, List[List[int]]]],
    action: torch.Tensor=None) -> dict:
        """
        Parameters
        ----------
        obs : list of 2-tuples of Pytorch Geometric Batch objects and list of lists of int
            Each tuple is a single observation (the entire list is a batch). Each Pytorch Geometric Batch object corresponds to
            the Pytorch Geometric graph representing the molecule. The list of lists of integers
            is a list of all the torsions of the molecule, where each torsion is represented by a list of four integers, where the integers
            are the indices of the four atoms making up the torsion.
        action : batch of torch.Tensor, optional
            If specified, the log probabilities returned by the network will be the log probabilities for the specified
            actions instead of for the newly sampled actions.

        Returns
        -------
        prediction : dict
            * prediction['v']: The value estimation.
            * prediction['a']: The action sampled from the distribution predicted by the network.
            * prediction['entropy'] The entropy of the distribution.
            * prediction['log_pi_a'] The log probabilities of the actions from the distribution.
        """
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

class _RTGNGatCritic(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, node_dim):
        super().__init__()
        self.gat = GAT(hidden_dim=hidden_dim, node_dim=node_dim)
        self.set2set = gnn.Set2Set(hidden_dim, processing_steps=6)
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
            )

    def forward(self, obs):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        out = self.gat(data)
        pool = self.set2set(out, data.batch)
        v = self.mlp(pool)

        return v

class _RTGNGatActor(torch.nn.Module):
    def __init__(self, action_dim, hidden_dim, node_dim):
        super().__init__()
        self.gat = self.gat = GAT(hidden_dim=hidden_dim, node_dim=node_dim)
        self.set2set = gnn.Set2Set(hidden_dim, processing_steps=6)

        self.fc = nn.Linear(2*hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(5*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

        self.hidden_dim = hidden_dim

    def forward(self, obs):
        data, nonring, nrbidx, torsion_list_sizes = obs
        N = data.num_graphs

        out = self.gat(data)
        pool = self.set2set(out, data.batch)
        graph_embed = self.fc(pool)

        graph_embed = torch.index_select(
            graph_embed,
            dim=0,
            index=nrbidx
        )

        graph_embed = graph_embed.view(-1, self.hidden_dim)

        out = torch.index_select(
            out,
            dim=0,
            index=nonring.view(-1)
        )
        out = out.view(-1, self.hidden_dim * 4)
        out = torch.cat([graph_embed,out],1)   # shape = (num_torsions, 5*self.hidden_dim)
        out = self.mlp(out)

        logit = out.split(torsion_list_sizes)
        logit = torch.nn.utils.rnn.pad_sequence(logit).permute(1, 0, 2)

        return logit
