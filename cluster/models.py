import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class CONVClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        self.convs.append(GraphConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(GraphConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)
    
    
class CONVRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        self.convs.append(GraphConv(in_channels, hidden_channels, aggr='mean'))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr='mean'))
        self.convs.append(GraphConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x.squeeze()