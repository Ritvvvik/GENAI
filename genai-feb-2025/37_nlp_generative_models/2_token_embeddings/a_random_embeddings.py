import torch
import torch.nn as nn

torch.manual_seed(123)
vocab_size, embed_dim = 10, 5
embed = nn.Embedding(vocab_size, embed_dim)
print(embed.state_dict())

inp_ids = torch.tensor([2])
output = embed(inp_ids)
print(output, output.shape)

inp_ids = torch.tensor([3])
output = embed(inp_ids)
print(output, output.shape)


inp_ids = torch.tensor([3, 2, 8])
output = embed(inp_ids)
print(output, output.shape)

inp_ids = torch.tensor([[3, 2, 8], [1, 4, 5]])
output = embed(inp_ids)
print(output, output.shape)
