import torch
import torch.nn as nn
from helper.set_handler import SetUnionHandler
import numpy as np
import math
torch.set_default_dtype(torch.float64)

class TransformerQNetworkNoEdges(nn.Module):
    def __init__(self, input_size, output_size, handler: SetUnionHandler, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(4, d_model, dtype=torch.float64)
        self.pos_encoding = nn.Parameter(torch.rand(1, input_size, d_model, dtype=torch.float64))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, 0.1, batch_first=True, dtype=torch.float64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.handler = handler

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        profits_norm = torch.tensor(self.handler.item_profits / self.handler.item_profits.max(), device=state.device)
        subset_sizes_norm = torch.tensor([len(sub) / self.handler.n for sub in self.handler.item_subsets], device=state.device)
        marginal_norm = torch.zeros(self.input_size, device=state.device)

        features = torch.stack([
            state.float(),
            profits_norm.expand(batch_size, -1),
            subset_sizes_norm.expand(batch_size, -1),
            marginal_norm.expand(batch_size, -1)
        ], dim=-1)
        x = self.embed(features) * torch.sqrt(torch.tensor([self.d_model], dtype=torch.float64))
        x = x + self.pos_encoding
        x = self.transformer(x)
        x_pooled = x.mean(dim=1)
        self.head(x_pooled)
        return self.head(x_pooled)