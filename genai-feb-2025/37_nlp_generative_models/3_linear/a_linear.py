import torch
import torch.nn as nn

torch.manual_seed(100)

in_features, out_features = 5, 2
lin = nn.Linear(in_features, out_features)
print(lin.state_dict())

input = torch.randn(in_features)
print(input, input.shape)
output = lin(input)
print(output, output.shape)

batch_size = 2
input = torch.randn(batch_size, in_features)
print(input, input.shape)
output = lin(input)
print(output, output.shape)

seq_len = 3
input = torch.randn(batch_size, seq_len, in_features)
print(input, input.shape)
output = lin(input)
print(output, output.shape)
