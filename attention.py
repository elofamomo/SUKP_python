import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, num_classes=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))  # Simplified positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]  # Add positional encoding
        output = self.transformer_encoder(src.transpose(0, 1), src_key_padding_mask=src_mask)
        output = output.mean(dim=0)  # Global average pooling
        output = self.fc(output)
        return output

# Example usage
x = torch.rand((1,201))
print(x)
print(x.squeeze(0).shape)