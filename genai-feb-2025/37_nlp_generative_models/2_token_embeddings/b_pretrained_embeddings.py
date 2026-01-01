import torch
import torch.nn as nn

torch.manual_seed(123)
vocab_size, embed_dim = 10, 5
weights = torch.randn(vocab_size, embed_dim)
embedding = nn.Embedding.from_pretrained(weights)
print(embedding, embedding.weight)

inp_ids = torch.tensor([2])
output = embedding(inp_ids)
print(output, output.shape)

inp_ids = torch.tensor([3])
output = embedding(inp_ids)
print(output, output.shape)


inp_ids = torch.tensor([3, 2, 8])
output = embedding(inp_ids)
print(output, output.shape)

inp_ids = torch.tensor([[3, 2, 8], [1, 4, 5]])
output = embedding(inp_ids)
print(output, output.shape)
