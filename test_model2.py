import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class PedestrianTrajectoryModel(nn.Module):
    def __init__(self, node_features, seq_length, num_nodes):
        super().__init__()
        self.gat = GATConv(in_channels=node_features, out_channels=node_features)
        self.gru = nn.GRU(input_size=node_features, hidden_size=node_features, batch_first=True)
        self.out = nn.Linear(in_features=node_features, out_features=2)  # x, y coordinates

    def forward(self, x, adj_matrix):
        batch_size, seq_length, num_nodes, num_features = x.size()
        # for t in range(seq_length):
        #     # Reshape x and adj_matrix for GAT input
        #     x_t = x[:, t, :, :].contiguous().view(-1, 4)  # Reshape to [batch_size * num_nodes, node_features]
        #     # adj_t = adj_matrix[:, t, :, :].contiguous().view(-1, 3)  # Reshape or process accordingly
        #     adj = adj_matrix[:, t, :, :]  # Reshape or process accordingly
        #     adj_t, _ = dense_to_sparse(adj)

        #     # print(x_t.shape, adj_t.shape)

        #     # Apply GAT for each time step
        #     gat_output = self.gat(x_t, adj_t)
        #     # Update features
        #     x[:, t, :, :] = gat_output.view(batch_size, num_nodes, -1)

        x = x.view(-1, num_nodes, num_features)
        edge_index = adj_matrix.view(2, -1)

        print(x.shape, edge_index.shape)
        gat_out = self.gat(x, edge_index)
        print(gat_out.shape)

        # Process the sequence of features through GRU
        # x = x.view(batch_size, seq_length, -1)
        # print(x.shape, x)
        # x = x.mean(dim=2)
        # print(x.shape, x)
        # gru_output, _ = self.gru(x)

        # print(gru_output.shape, gru_output)

        # Apply the output layer to each time step
        # predictions = self.out(gru_output)
        # predictions = predictions.view(batch_size, -1, num_nodes, 2)  # Reshape to the desired output

        return predictions

# Initialize the model
model = PedestrianTrajectoryModel(node_features=4, seq_length=8, num_nodes=3)

# Dummy data
x = torch.rand(1, 8, 3, 4)  # batch_size, seq_length, num_nodes, node_features
adj_matrix = torch.rand(1, 8, 3, 3)  # batch_size, seq_length, num_nodes, num_nodes

# Forward pass
output = model(x, adj_matrix)
print(output.shape)  # Should be torch.Size([1, 12, 3, 2])
