import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
import torch.nn.functional as F

class GraphConvModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_layers, mode='sum'):
        super().__init__()
        self.num_layers = num_layers
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * (num_layers - 1)
        else:
            hidden_dims = hidden_dim
        assert len(hidden_dims) == num_layers - 1, "Length of hidden_dims must be num_layers - 1"

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_dims[0], aggr=mode))
        for i in range(1, num_layers - 1):
            self.convs.append(GraphConv(hidden_dims[i - 1], hidden_dims[i], aggr=mode))
        self.convs.append(GraphConv(hidden_dims[-1], num_classes, aggr=mode))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x
    
