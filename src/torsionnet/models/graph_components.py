from torch import nn
import torch.nn.functional as F
import torch
import torch_geometric.nn as gnn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MPNN(torch.nn.Module):
    def __init__(self, hidden_dim, edge_dim, node_dim, message_passing_steps=6):
        super().__init__()
        self.mpnn_iters = message_passing_steps
        self.fc = torch.nn.Linear(node_dim, hidden_dim)
        func_ag = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(inplace=False), nn.Linear(hidden_dim, hidden_dim * hidden_dim))
        self.conv = gnn.NNConv(hidden_dim, hidden_dim, func_ag, aggr='mean')
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, data):
        out = F.relu(self.fc(data.x))
        h = out.unsqueeze(0)

        for i in range(self.mpnn_iters):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        return out

class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, node_dim, num_layers=6):
        super().__init__()
        self.fc = torch.nn.Linear(node_dim, hidden_dim)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(gnn.GATConv(hidden_dim, hidden_dim, heads=2))
        for i in range(num_layers - 2):
            self.conv_layers.append(gnn.GATConv(hidden_dim*2, hidden_dim, heads=2))
        self.conv_layers.append(gnn.GATConv(hidden_dim*2, hidden_dim))

    def forward(self, data):
        out = F.relu(self.fc(data.x))
        for layer in self.conv_layers:
            out = layer(out, data.edge_index)
        return out