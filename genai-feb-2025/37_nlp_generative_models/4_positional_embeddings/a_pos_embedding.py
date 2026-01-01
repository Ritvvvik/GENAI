import torch
from torch.nn import Embedding

torch.manual_seed(123)
context_length, embed_dim = 7, 5
pos_embed = Embedding(context_length, embed_dim)
print(pos_embed.state_dict())

positions = torch.arange(0, context_length).unsqueeze(0)
print(positions)
output = pos_embed(positions)
print(output, output.shape)

positions = torch.arange(0, 4).unsqueeze(0)
print(positions)
output = pos_embed(positions)
print(output, output.shape)
