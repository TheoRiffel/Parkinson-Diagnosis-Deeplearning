import torch.nn as nn

class customMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(customMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)