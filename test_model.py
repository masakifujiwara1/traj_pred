import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class PedestrianTrajectoryModel(nn.Module):
    def __init__(self, node_features, seq_length, num_nodes, num_heads):
        super(PedestrianTrajectoryModel, self).__init__()
        self.num_heads = num_heads
        self.gat = GATConv(in_channels=node_features, out_channels=64, heads=num_heads)
        self.gru = nn.GRU(input_size=64, hidden_size=node_features, batch_first=True)
        self.out = nn.Linear(in_features=4, out_features=2)  # x, y coordinates
        self.conv1 = nn.Conv2d(8, 12, 3, padding=1)

    def forward(self, x, adj_matrix):
        batch_size, seq_length, num_nodes, num_features = x.size()
        GAT_out = torch.empty(batch_size, seq_length, num_nodes, 64)
        for t in range(seq_length):
            # Reshape x and adj_matrix for GAT input
            x_t = x[:, t, :, :].contiguous().view(-1, 4)  # Reshape to [batch_size * num_nodes, node_features]
            # adj_t = adj_matrix[:, t, :, :].contiguous().view(-1, 3)  # Reshape or process accordingly
            adj = adj_matrix[:, t, :, :]  # Reshape or process accordingly
            adj_t, _ = dense_to_sparse(adj)
            # adj_t = adj.reshape(-1, adj.size(-1))

            # print(adj_t.shape)
            # print(x_t.shape)

            # Apply GAT for each time step
            gat_output = self.gat(x_t, adj_t)
            # print(gat_output.shape)
            # Update features
            # print(x.shape)
            # x[:, t, :, :] = gat_output.view(batch_size, num_nodes, -1)
            GAT_out[:, t, :, :] = gat_output.view(batch_size, num_nodes, -1)
        
        print("GAT_OUT", GAT_out.shape)

        # Process the sequence of features through GRU
        # x = x.view(batch_size, seq_length, -1)
        # print(x.shape, x)
        # x = x.mean(dim=2)

        # x = x.view(batch_size*num_nodes, seq_length, num_features)
        GAT_out = GAT_out.view(batch_size*num_nodes, seq_length, 64)
        print(GAT_out.shape)

        # print(x.shape, x)
        gru_output, _ = self.gru(GAT_out)

        gru_output = gru_output.view(batch_size, num_nodes, seq_length, num_features)

        # print(gru_output.shape, gru_output)

        gru_output = gru_output.permute(0, 2, 1, 3)

        gru_output = self.conv1(gru_output)
        gru_output = gru_output.view(-1, 4)
        print(gru_output.shape)
        gru_output = self.out(gru_output)
        gru_output = gru_output.view(batch_size, 12, num_nodes, 2)
        # Apply the output layer to each time step
        # predictions = self.out(gru_output)
        # predictions = predictions.view(batch_size, -1, num_nodes, 2)  # Reshape to the desired output

        return gru_output

# Initialize the model
model = PedestrianTrajectoryModel(node_features=4, seq_length=8, num_nodes=3, num_heads=1)

# Dummy data
x = torch.rand(1, 8, 3, 4)  # batch_size, seq_length, num_nodes, node_features
adj_matrix = torch.rand(1, 8, 3, 3)  # batch_size, seq_length, num_nodes, num_nodes

# Forward pass
output = model(x, adj_matrix)
print("1 :", output.shape)  # Should be torch.Size([1, 12, 3, 2])

# Dummy data
x = torch.rand(1, 8, 16, 4)  # batch_size, seq_length, num_nodes, node_features
adj_matrix = torch.rand(1, 8, 16, 16)  # batch_size, seq_length, num_nodes, num_nodes

# Forward pass
output = model(x, adj_matrix)
print("2 :", output.shape)  # Should be torch.Size([1, 12, 3, 2])
