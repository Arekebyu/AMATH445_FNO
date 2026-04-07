import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    """
    Fully Convolutional Network (FCN).
    Operates on grid data using N-dimensional convolutions.
    Expects input shape: (batch, ..., channels) and internally transposes.
    Supports 1D, 2D, and 3D grids.
    """
    def __init__(self, in_channels, out_channels, n_dims=1, hidden_channels=64, num_layers=4):
        super(FCN, self).__init__()
        
        self.n_dims = n_dims
        
        if n_dims == 1:
            Conv = nn.Conv1d
        elif n_dims == 2:
            Conv = nn.Conv2d
        elif n_dims == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"FCN only supports 1, 2, or 3 dimensions. Got {n_dims}.")
            
        layers = []
        layers.append(Conv(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(Conv(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            
        layers.append(Conv(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: (batch, x_1, x_2, ..., x_n, in_channels)
        # PyTorch ConvNd expects shape: (batch, in_channels, x_1, x_2, ..., x_n)
        
        # Move channel dimension from last to second position
        dims = list(range(x.dim()))
        dims = [dims[0], dims[-1]] + dims[1:-1]
        x = x.permute(dims)
        
        x = self.net(x)
        
        # Move channel dimension back to last position
        inv_dims = [0] + list(range(2, x.dim())) + [1]
        x = x.permute(inv_dims)
        
        return x
