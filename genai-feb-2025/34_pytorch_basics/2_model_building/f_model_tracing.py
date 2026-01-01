# pip install torchinfo
import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F


class RegressionModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim1)
        self.hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        print(x, x.shape)
        x = F.relu(self.input(x))
        print(x, x.shape)
        x = F.relu(self.hidden(x))
        print(x, x.shape)
        x = self.output(x)
        print(x, x.shape)
        return x


if __name__ == "__main__":
    model = RegressionModel1(1, 10, 5, 1)
    print(model)
    # model inference
    inp = torch.tensor([0.1])
    print(model(inp))

    # model weights (all)
    print(model.state_dict())

    # model weights (iterative)
    for name, param in model.named_parameters():
        print(name, param.data.shape, param.data)
