import torch
import torch.nn as nn

class DeepQlearningNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQlearningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),  # Wider first layer for larger m
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),  # Second layer
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),  # Third layer
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),  # Added fourth layer for depth
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Fifth layer
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_size)  # Output
        )

    def forward(self, x):
        return self.network(x)