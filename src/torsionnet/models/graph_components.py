from torch import nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

import logging
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MPNN(torch.nn.Module):
    def __init__(self, edge_dim, dim, num_features=3, message_passing_steps=6):
        super().__init__()
        self.mpnn_iters = message_passing_steps
        self.fc = torch.nn.Linear(num_features, dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, dim), nn.ReLU(inplace=False), nn.Linear(dim, dim * dim))
        self.conv = gnn.NNConv(dim, dim, func_ag, aggr='mean')
        self.gru = nn.GRU(dim, dim)

    def forward(self, data):
        out = F.relu(self.fc(data.x))
        h = out.unsqueeze(0)

        for i in range(self.mpnn_iters):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        return out