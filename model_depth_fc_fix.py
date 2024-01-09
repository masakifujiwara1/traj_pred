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
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.fc1 = nn.Linear(in_features, 16)
        self.fc_seq = nn.Linear(8, 12)
        self.gat1 = GAT_Layer(30, 30, num_heads)
        self.gat2 = GAT_Layer(30, 30, num_heads=1)
        self.gru1 = nn.GRU(input_size=16, hidden_size=30, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=30, hidden_size=30, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(hidden_features, hidden_features, batch_first=True)
        self.out1 = nn.Linear(in_features=30, out_features=hidden_features)
        self.out2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.conv = nn.Conv2d(obs_seq_len, pred_seq_len, 3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 30)
        )
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x, adj_matrix):
        batch_size, seq_len, num_nodes, num_features = x.size()

        # embedding
        # [B, S, N, F] >> [N, S, F]
        x_emb = x.permute(0, 2, 1, 3).squeeze()
        # each node encode
        processed_traj = [self.relu(self.fc1(traj)) for traj in x_emb]

        processed_gru = []
        for emb_in in processed_traj:
            emb_out, _ = self.gru1(emb_in)
            processed_gru.append(self.relu(emb_out))
        x_gru = torch.stack(processed_gru, dim=0) # [N, S, F]
        x_gru = x_gru.permute(1, 0, 2) # [S, N, F]
        x_gru = x_gru.view(1, seq_len, num_nodes, 30) # [B, S, N, F]

        x1 = self.gat1(x_gru, adj_matrix)

        # GAT layer
        x2 = self.relu(x1)
        # x2 = self.dropout(x2)
        # skip connection
        x2 = x2 + x_gru

        x3 = self.relu(self.mlp(x2)) # [1, 8, 3, 30]
        x3 = x3.permute(0, 2, 1, 3).squeeze()

        processed_gru2 = []
        for gru_in in x3:
            gru_out, _ = self.gru2(gru_in)
            processed_gru2.append(self.relu(gru_out))
        x4 = torch.stack(processed_gru2, dim=0)

        x4 = x4.permute(1, 0, 2) # [S, N, F]
        x5 = self.conv(x4) # [12, 3, 30]

        # x4 = x4.permute(0, 2, 1)
        # x5 = self.fc_seq(x4)
        x6 = self.relu(x5)
        # print(x6.shape)
        # x6 = x6.permute(2, 0, 1)

        # FC layer
        x8 = self.out1(x6)
        x9 = self.relu(x8)
        x10 = self.out2(x9)
        x10 = x10.view(batch_size, self.pred_seq_len, num_nodes, self.out_features)

        return x10

class GAT_Layer(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads):
        super(GAT_Layer, self).__init__()
        self.hidden_features = hidden_features
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
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
# model = GAT_TimeSeriesLayer(in_features=4, hidden_features=16, out_features=5, obs_seq_len=8, pred_seq_len=12, num_heads=1).cuda()

# # Dummy data
# x = torch.rand(1, 8, 3, 4).cuda()  # batch_size, seq_length, num_nodes, node_features
# adj_matrix = torch.rand(1, 8, 3, 3).cuda()  # batch_size, seq_length, num_nodes, num_nodes

# # Forward pass
# output = model(x, adj_matrix)
# print("1 :", output.shape, output.device)  # Should be torch.Size([1, 12, 3, 2])