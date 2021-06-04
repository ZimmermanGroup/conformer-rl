"""
Graph_components
================
Modularized graph neural network components using PyTorch Geometric.
"""
from torch import nn
import torch.nn.functional as F
import torch
import torch_geometric.nn as gnn
from torch_geometric.data import Batch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MPNN(torch.nn.Module):
    """
    Implements a basic unit of the message passing neural network (MPNN) [1]_.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layer.
    edge_dim : int
        Dimension of the edge embeddings in the input graph.
    node_dim : int
        Dimension of the node embeddings in the input graph.
    message_passing_steps : int 
        Number of message passing steps to execute. See [1]_ for more details.

    References
    ----------
    .. [1] `MPNN paper <https://arxiv.org/abs/1704.01212>`_
    """
    def __init__(self, hidden_dim: int, edge_dim: int, node_dim: int, message_passing_steps: int=6):
        super().__init__()
        self.mpnn_iters = message_passing_steps
        self.fc = torch.nn.Linear(node_dim, hidden_dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(inplace=False), nn.Linear(hidden_dim, hidden_dim * hidden_dim))
        self.conv = gnn.NNConv(hidden_dim, hidden_dim, func_ag, aggr='mean')
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, data: Batch) -> torch.Tensor:
        out = F.relu(self.fc(data.x))
        h = out.unsqueeze(0)

        for i in range(self.mpnn_iters):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return out

class GAT(torch.nn.Module):
    """
    Implements a basic unit of the graph attention network (GAT) [2]_.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden layer.
    node_dim : int
        Dimension of the node embeddings in the input graph.
    num_layers : int 
        Number of GAT conv layers. See [2]_ for more details.

    References
    ----------
    .. [2] `GAT paper <https://arxiv.org/abs/1710.10903>`_
    """
    def __init__(self, hidden_dim: int, node_dim: int, num_layers: int=6):
        super().__init__()
        self.fc = torch.nn.Linear(node_dim, hidden_dim)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(gnn.GATConv(hidden_dim, hidden_dim, heads=2))
        for i in range(num_layers - 2):
            self.conv_layers.append(gnn.GATConv(hidden_dim*2, hidden_dim, heads=2))
        self.conv_layers.append(gnn.GATConv(hidden_dim*2, hidden_dim))

    def forward(self, data : Batch) -> torch.Tensor:
        out = F.relu(self.fc(data.x))
        for layer in self.conv_layers:
            out = layer(out, data.edge_index)
        return out