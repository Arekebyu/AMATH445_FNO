import torch
import torch.nn as nn
from SpectralConv import SpectralConv

class FNOBlock(nn.Module):
    def __init__(self, width, modes, n_dims):
        super(FNOBlock, self).__init__()
        self.K = SpectralConv(width, width, modes, n_dims)
        if n_dims == 1:
            self.W = nn.Conv1d(width, width, 1)
        elif n_dims == 2:
            self.W = nn.Conv2d(width, width, 1)
        elif n_dims == 3:
            self.W = nn.Conv3d(width, width, 1)
    
    def forward(self, x):
        res = self.W(x)
        spec = self.K(x)
        return torch.nn.functional.gelu(res + spec)
