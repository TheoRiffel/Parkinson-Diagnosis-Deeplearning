import torch.nn as nn
from torch.nn.functional import avg_pool1d

class customFCN(nn.Module):
  def __init__(self, input_channels, n_classes):
    
    super(customFCN, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv1d(in_channels=input_channels, 
                  out_channels=128,
                  kernel_size=64, 
                  padding='same'),  
        nn.BatchNorm1d(128),  
        nn.ReLU(),
        # nn.Dropout(0.5)
    )
    
    self.block2 = nn.Sequential(
        nn.Conv1d(in_channels=128, 
                  out_channels=256, 
                  kernel_size=16,
                  padding='same'),  
        nn.BatchNorm1d(256),  
        nn.ReLU(),
        # nn.Dropout(0.5)
    )
    
    self.block3 = nn.Sequential(
        nn.Conv1d(in_channels=256, 
                  out_channels=128, 
                  kernel_size=16,
                  padding='same'),  
        nn.BatchNorm1d(128),  
        nn.ReLU(),
        # nn.Dropout(0.5)
    )
    
    self.fc = nn.Sequential(
        nn.Linear(128, n_classes),  
    )
  
  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = avg_pool1d(x, x.shape[-1])
    x = x.squeeze(-1)
    x = self.fc(x)
    return x