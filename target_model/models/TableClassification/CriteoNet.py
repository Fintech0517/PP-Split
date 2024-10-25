import torch
import torch.nn as nn

class FirstNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FirstNet, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = self.L1(x)
        x = nn.functional.relu(x)
        return x


class SecondNet(nn.Module):
    def __init__(self,hidden_dim):
        super(SecondNet, self).__init__()
        self.L2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.L2(x)
        x = torch.sigmoid(x)
        return x