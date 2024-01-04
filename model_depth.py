import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class GAT_TimeSeriesLayer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, obs_seq_len, pred_seq_len, num_heads):
        super(GAT_TimeSeriesLayer, self).__init__()
        self.pred_seq_len = pred_seq_len
        self.out_features = out_features
        self.hidden_features = hidden_features
        # self.gat = GATConv(in_channels=in_features, out_channels=hidden_features, heads=num_heads)
        self.gat1 = GAT_Layer(in_features, hidden_features, num_heads)
        self.gat2 = GAT_Layer(hidden_features*num_heads, hidden_features, num_heads=1)
        self.gru = nn.GRU(input_size=hidden_features, hidden_size=hidden_features, num_layers=2, batch_first=True)
        self.out1 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.out2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.conv = nn.Conv2d(obs_seq_len, pred_seq_len, 3, padding=1)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.prelu = nn.PReLU()

    def forward(self, x, adj_matrix):
        batch_size, seq_len, num_nodes, num_features = x.size()
        x1 = torch.empty(batch_size, seq_len, num_nodes, self.hidden_features).to(self.device)
        x3 = torch.empty(batch_size, seq_len, num_nodes, self.hidden_features).to(self.device)

        # GAT layer
        x1 = self.gat1(x, adj_matrix)
        x2 = self.prelu(x1)
        x3 = self.gat2(x2, adj_matrix)
        x4 = self.prelu(x3)

        x4 = x4.view(batch_size*num_nodes, seq_len, self.hidden_features)

        # GRU layer (num_layer = 2)
        x5, _ = self.gru(x4)
        x5 = x5.view(batch_size, num_nodes, seq_len, self.hidden_features)
        x5 = x5.permute(0, 2, 1, 3)

        # Conv Layer
        x6 = self.conv(x5)
        x7 = self.prelu(x6)
        x7 = x7.view(-1, self.hidden_features)

        # FC layer
        x8 = self.out1(x7)
        x9 = self.prelu(x8)
        x10 = self.out2(x9)
        x10 = x10.view(batch_size, self.pred_seq_len, num_nodes, self.out_features)

        return x10

class GAT_Layer(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads):
        super(GAT_Layer, self).__init__()
        self.gat = GATConv(in_channels=in_features, out_channels=hidden_features, heads=num_heads)

    def forward(self, x, adj_matrix):
        batch_size, seq_len, num_nodes, num_features = x.size()
        gat_output_reshaped = torch.empty(batch_size, seq_len, num_nodes, self.hidden_features).to(self.device)

        for time_step in range(seq_len):
            x_t = x[:, time_step, :, :].contiguous().view(-1, num_features)
            adj = adj_matrix[:, time_step, :, :]
            adj_t, _ = dense_to_sparse(adj)
            gat_output = self.gat(x_t, adj_t)
            gat_output_reshaped[:, time_step, :, :] = gat_output.view(batch_size, num_nodes, -1)

        return gat_output_reshaped
# debug
# Initialize the model
# model = GAT_TimeSeriesLayer(in_features=4, hidden_features=64, out_features=2, obs_seq_len=8, pred_seq_len=12, num_heads=1).cuda()

# # Dummy data
# x = torch.rand(1, 8, 3, 4).cuda()  # batch_size, seq_length, num_nodes, node_features
# adj_matrix = torch.rand(1, 8, 3, 3).cuda()  # batch_size, seq_length, num_nodes, num_nodes

# # Forward pass
# output = model(x, adj_matrix)
# print("1 :", output.shape, output.device)  # Should be torch.Size([1, 12, 3, 2])