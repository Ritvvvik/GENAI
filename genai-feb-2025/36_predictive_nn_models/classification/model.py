import torch.nn as nn
import torch.nn.functional as F
import torch


class CreditScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.softmax(self.output_layer(x), dim=0)
        return x


class CreditScoreModel2(nn.Module):
    def __init__(
        self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim1)
        self.hidden_layer_1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden_layer_2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.hidden_layer_3 = nn.Linear(hidden_dim3, hidden_dim4)
        self.output_layer = nn.Linear(hidden_dim4, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))
        x = F.relu(self.hidden_layer_3(x))
        x = F.softmax(self.output_layer(x), dim=0)
        return x


if __name__ == "__main__":
    inp = torch.randn(29)
    model = CreditScoreModel2(29, 50, 25, 10, 5, 3)
    res = model(inp)
    print(res)
    print(torch.argmax(res))
