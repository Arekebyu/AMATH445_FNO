import torch
import torch.nn as nn
from .SpectralConv import SpectralConv

class FNOBlock(nn.Module):
    def __init__(self, width, modes, n_dims):
        super(FNOBlock, self).__init__()
        self.K = SpectralConv(width, width, modes, n_dims)
        self.W = nn.Conv2d(width, width, 1)
    
    def forward(self, x):
        res = self.W(x)
        spec = self.K(x)
        return torch.nn.functional.gelu(res + spec)
