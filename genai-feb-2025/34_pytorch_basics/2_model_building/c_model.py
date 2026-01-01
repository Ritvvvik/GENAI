# pip install torchinfo
import torch.nn as nn
import torch
from torchinfo import summary


class RegressionModel1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":
    model = RegressionModel1(1, 1)
    # model inference
    inp = torch.tensor([0.1])
    print(model(inp))
    # model.__call__(inp)
    # preprocess
    # self.forward(inp)
    # postprocess
    print(model)

    # model info(external library)
    summary(model)

    # model weights (all)
    print(model.state_dict())

    # model weights (iterative)
    for name, param in model.named_parameters():
        print(name, param.data.shape, param.data)
