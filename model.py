import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT_GRU(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super(GAT_GRU, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=4)
        self.gru = nn.GRU(out_channels, hidden_channels, num_layers=num_layers)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.permute(1, 2, 0)
        x, _ = self.gru(x)
        x = x.permute(2, 0, 1)
        return x