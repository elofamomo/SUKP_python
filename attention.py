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
vocab_size = 10000  # Assume a vocabulary size
model = SimpleTransformer(vocab_size)

# Dummy input: batch of 2 sequences, each of length 5 (token IDs)
input_seq = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
output = model(input_seq)
print("Model Output:\n", output)