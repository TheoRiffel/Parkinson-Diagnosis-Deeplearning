import torch.nn as nn

class tsMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(tsMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # aumente o dropout
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)