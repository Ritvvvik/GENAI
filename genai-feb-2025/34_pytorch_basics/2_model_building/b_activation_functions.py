import torch
import torch.nn.functional as F

data = torch.randn(2)
print(data)
print(F.relu(data))

data = torch.randn(4)
print(data)
print(F.relu(data))

data = torch.rand(5)
print(data)
print(F.softmax(data, dim=0))
