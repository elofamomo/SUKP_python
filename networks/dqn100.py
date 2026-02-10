import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool  # Requires torch-geometric
import torch_geometric.data as data


class DeepQlearningNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQlearningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),  # First hidden layer
            nn.LayerNorm(2048),  # Normalize activations
            nn.ReLU(),
            nn.Linear(2048, 1024),  # Second hidden layer
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),  # Third hidden layer
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Fourth hidden layer
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_size)  # Output: No norm or activation for Q-values
        )

    def forward(self, x):
        return self.network(x)
