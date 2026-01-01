import torch

# Create a torch.Tensor object with the given data.  It is a 1D vector
V_data = [1.0, 2.0, 3.0]
V = torch.Tensor(V_data)
print(type(V))
print(V)

# Creates a matrix
M_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6]]
M = torch.Tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2.
T_data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
T = torch.Tensor(T_data)
print(T)

# Create a random tensors
torch.manual_seed(100)
y = torch.randn(2)
print(y)

x = torch.randn(3, 4)
print(x)


# Create special tensors
z = torch.ones(2, 2)
print(z)
