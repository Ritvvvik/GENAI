import torch
import torch.nn as nn

torch.manual_seed(100)

batch_size, seq_length, embed_dim = 2, 5, 10
layer_norm = nn.LayerNorm(embed_dim)
print(layer_norm.state_dict())

input = torch.randn(batch_size, seq_length, embed_dim)
print(input, input.shape)
output = layer_norm(input)
print(output, output.shape)
