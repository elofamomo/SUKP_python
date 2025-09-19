import torch
import torch.nn as nn
import numpy as np
import math

class DeepQlearningNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQlearningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),  # First hidden layer
            nn.LayerNorm(1024),  # Normalize activations
            nn.ReLU(),
            nn.Linear(1024, 512),  # Second hidden layer
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Third hidden layer
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Fourth hidden layer
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_size)  # Output: No norm or activation for Q-values
        )

    def forward(self, x):
        return self.network(x)
    
class TransformerQNetwork(nn.Module):
    def __init__(self, input_size, output_size, profits, subset_size, d_model=128, nhead=8, num_layers=4):
        """
        Transformer-based Q-Network for SUKP DQN.
        
        :param input_size: int, m (number of items)
        :param output_size: int, 2*m + 1 (actions)
        :param d_model: int, embedding dimension
        :param nhead: int, attention heads (divides d_model)
        :param num_layers: int, encoder layers
        """
        super(TransformerQNetwork, self).__init__()
        self.input_size = input_size  # m
        self.d_model = d_model
        self.feature_dim = 3  # selected, profit_norm, subset_size_norm
        self.profits = profits
        self.subset_size = subset_size
        # Embed item features to d_model
        self.item_embed = nn.Linear(self.feature_dim, d_model)
        
        # Learned positional encoding for item positions
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pool (mean) and project to Q-values
        self.pool_proj = nn.Linear(d_model, output_size)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embed.weight)
        nn.init.xavier_uniform_(self.pool_proj.weight)

    def forward(self, state):
        """
        Forward pass for SUKP state.
        
        :param state: torch.Tensor, (batch, m), binary selection
        :param profits: torch.Tensor, (m,), item profits (normalized)
        :param subset_sizes: torch.Tensor, (m,), subset sizes (normalized)
        :return: torch.Tensor, (batch, output_size), Q-values
        """
        profits = self.profits
        subset_sizes = self.subset_size
        if state.dim() == 1:
            state = state.unsqueeze(0)
            profits = profits.unsqueeze(0)
            subset_sizes = subset_sizes.unsqueeze(0)
        batch_size = state.shape[0]
        # Item features: (batch, m, feature_dim)
        features = torch.stack([
            state.float(),  # Selected (0/1)
            profits.expand(batch_size, -1),  # Normalized profit
            subset_sizes.expand(batch_size, -1)  # Normalized subset size (proxy for overlap potential)
        ], dim=-1)
        
        # Embed: (batch, m, d_model)
        x = self.item_embed(features) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :self.input_size]  # Add positional
        
        # Self-attention: (batch, m, d_model)
        x = self.transformer(x)
        
        # Global average pool over items: (batch, d_model)
        x_pooled = x.mean(dim=1)
        
        # Project to Q-values: (batch, output_size)
        q_values = self.pool_proj(x_pooled)
        return q_values

