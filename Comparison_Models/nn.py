import torch
import torch.nn as nn

class NN(nn.Module):
    """
    Point-wise Feedforward Neural Network.
    Applies a simple fully-connected network to each spatial point independently.
    Expects input of shape: (batch, ..., channels).
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4):
        super(NN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch, ..., in_channels)
        # Linear layer naturally operates on the last dimension
        return self.net(x)
