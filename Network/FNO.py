import torch
import torch.nn as nn
from FNOBlock import FNOBlock

class FNO(nn.Module):
    def __init__(self, modes, input_size, output_size, n_dims, num_layers, width):
        super(FNO, self).__init__()
        self.n_dims = n_dims
        self.T = num_layers

        self.shallow = nn.Linear(input_size, width)
        self.FNOLayers = nn.ModuleList([FNOBlock(width, modes, n_dims) for _ in range(num_layers)])
        
        # Standard projection layer logic for FNO
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x is (Batch, Dim1, Dim2..., InputChannels)
        x = self.shallow(x)
        
        # Permute for Convolution layers: (Batch, InputChannels, Dim1, Dim2...)
        dims = list(range(1, self.n_dims + 2))
        x = x.permute(0, dims[-1], *dims[:-1])
        
        for t in range(self.T):
            x = self.FNOLayers[t](x)
            
        # Permute back: (Batch, Dim1, Dim2..., InputChannels)
        inv_dims = list(range(2, self.n_dims + 2)) + [1]
        x = x.permute(0, *inv_dims)
        
        x = self.projection(x)
        return x