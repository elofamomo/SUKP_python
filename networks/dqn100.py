import torch
import torch.nn as nn

class DeepQlearningNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQlearningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),     # First hidden layer
            nn.ReLU(),
            nn.Linear(1024, 512),            # Second hidden layer
            nn.ReLU(),
            nn.Linear(512, 256),            # Third hidden layer
            nn.ReLU(),
            nn.Linear(256, 256),            # Third hidden layer
            nn.ReLU(),
            nn.Linear(256, output_size)     # Output: 2*m+1 (e.g., 201 for m=100)
        )

    def forward(self, x):
        return self.network(x)
