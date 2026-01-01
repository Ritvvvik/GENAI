import torch
import torch.nn as nn

torch.manual_seed(100)
data = torch.randn(5)
print(data)

# single neuron layer
lin = nn.Linear(5, 1)
print(lin.weight)
print(lin(data))

# two neuron layer
lin = nn.Linear(5, 2)
print(lin.weight)
print(lin(data))
