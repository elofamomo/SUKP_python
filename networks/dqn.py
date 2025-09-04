import torch
import torch.nn as nn

class DeepQlearningNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQlearningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),  # First hidden layer
            nn.LayerNorm(256),         # Batch normalization
            nn.ReLU(),
            nn.Linear(256, 256),         # Second hidden layer
            nn.LayerNorm(256),         # Batch normalization
            nn.ReLU(),
            nn.Linear(256, 128),         # Third hidden layer
            nn.LayerNorm(128),         # Batch normalization
            nn.ReLU(),
            nn.Linear(128, output_size)  # Output layer for Q-values
        )

    def forward(self, x):
        return self.network(x)
