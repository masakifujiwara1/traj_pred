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
        self.gat = GATConv(in_channels=in_features, out_channels=hidden_features, heads=num_heads)
        self.gru = nn.GRU(input_size=hidden_features, hidden_size=hidden_features, batch_first=True)
        self.out = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.conv = nn.Conv2d(obs_seq_len, pred_seq_len, 3, padding=1)

    def forward(self, x, adj_matrix):
        batch_size, seq_len, num_nodes, num_features = x.size()
        gat_output_reshaped = torch.empty(batch_size, seq_len, num_nodes, self.hidden_features)

        for time_step in range(seq_len):
            x_t = x[:, time_step, :, :].contiguous().view(-1, num_features)
            adj = adj_matrix[:, time_step, :, :]
            adj_t, _ = dense_to_sparse(adj)
            gat_output = self.gat(x_t, adj_t)
            gat_output_reshaped[:, time_step, :, :] = gat_output.view(batch_size, num_nodes, -1)

        gat_output_reshaped = gat_output_reshaped.view(batch_size*num_nodes, seq_len, self.hidden_features)

        gru_output, _ = self.gru(gat_output_reshaped)
        gru_output_reshaped = gru_output.view(batch_size, num_nodes, seq_len, self.hidden_features)
        gru_output_reshaped = gru_output_reshaped.permute(0, 2, 1, 3)

        conv_output = self.conv(gru_output_reshaped)
        conv_output = conv_output.view(-1, self.hidden_features)
        out = self.out(conv_output)
        out_reshaped = out.view(batch_size, self.pred_seq_len, num_nodes, self.out_features)

        return out_reshaped

# Initialize the model
model = GAT_TimeSeriesLayer(in_features=4, hidden_features=64, out_features=2, obs_seq_len=8, pred_seq_len=12, num_heads=1)

# Dummy data
x = torch.rand(1, 8, 3, 4)  # batch_size, seq_length, num_nodes, node_features
adj_matrix = torch.rand(1, 8, 3, 3)  # batch_size, seq_length, num_nodes, num_nodes

# Forward pass
output = model(x, adj_matrix)
print("1 :", output.shape)  # Should be torch.Size([1, 12, 3, 2])