import torch
import torch.nn as nn
import numpy as np

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


# class DeepQlearningNetwork(nn.Module):
#     def __init__(self, input_size, output_size, d_model=128, nhead=4, num_layers=2, feature_dim=3):
#         """
#         Transformer-based Q-Network for SUKP DQN.
        
#         :param input_size: int, number of items (m, for binary state vector)
#         :param output_size: int, number of actions (2*m + 1 for add, remove, terminate)
#         :param d_model: int, embedding dimension for transformer
#         :param nhead: int, number of attention heads (must divide d_model)
#         :param num_layers: int, number of transformer encoder layers
#         :param feature_dim: int, number of features per item (e.g., selected, profit, marginal weight)
#         """
#         super(DeepQlearningNetwork, self).__init__()
#         self.input_size = input_size  # m
#         self.d_model = d_model
#         self.feature_dim = feature_dim
        
#         # Embedding layer: map feature_dim (e.g., 3) to d_model
#         self.embed = nn.Linear(feature_dim, d_model)
        
#         # Positional encoding (learned)
#         self.pos_encoder = nn.Parameter(torch.zeros(1, input_size, d_model))
        
#         # Transformer encoder
#         encoder_layers = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 4,  # Standard FFN size
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
#         # Output head: flatten transformer output and map to Q-values
#         self.fc_out = nn.Linear(d_model * input_size, output_size)
        
#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         """Initialize weights for stability."""
#         nn.init.xavier_uniform_(self.embed.weight)
#         nn.init.xavier_uniform_(self.fc_out.weight)
#         nn.init.zeros_(self.fc_out.bias)

#     def forward(self, state, handler=None):
#         """
#         Forward pass: Process state to output Q-values.
        
#         :param state: torch.Tensor, shape (batch, input_size), binary selection vector
#         :param handler: SetUnionHandler, optional for computing item features
#         :return: torch.Tensor, shape (batch, output_size), Q-values for actions
#         """
#         if state.dim() == 1:
#             state = state.unsqueeze(0)  # Add batch dim: (m) -> (1, m)
        
#         batch_size = state.shape[0]
        
#         # Create item features: (batch, m, feature_dim)
#         # Features: [selected (0/1), normalized profit, normalized marginal weight]
#         if handler is not None:
#             profits = torch.tensor(handler.item_profits, dtype=torch.float32, device=state.device) / handler.item_profits.max()
#             marginal_weights = torch.zeros(self.input_size, device=state.device)
#             for i in range(self.input_size):
#                 marginal = sum(handler.element_weights[e] for e in handler.item_subsets[i] if handler.element_counts[e] == 0)
#                 marginal_weights[i] = marginal
#             marginal_weights = marginal_weights / (handler.element_weights.max() or 1.0)
#             features = torch.stack([
#                 state,  # Binary selection (0/1)
#                 profits.expand(batch_size, -1),  # Normalized profit
#                 marginal_weights.expand(batch_size, -1)  # Normalized marginal weight
#             ], dim=-1)  # Shape: (batch, m, 3)
#         else:
#             # Fallback: Only use binary state
#             features = state.unsqueeze(-1)  # (batch, m, 1)
        
#         # Embed features: (batch, m, feature_dim) -> (batch, m, d_model)
#         print(self.embed(features))
#         x = self.embed(features) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
#         x = x + self.pos_encoder  # Add positional encoding
        
#         # Transformer encoder: (batch, m, d_model) -> (batch, m, d_model)
#         x = self.transformer_encoder(x)
        
#         # Flatten and project to Q-values: (batch, m*d_model) -> (batch, output_size)
#         x = x.flatten(start_dim=1)
#         return self.fc_out(x)