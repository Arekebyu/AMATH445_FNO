import torch
import torch.nn as nn
from FNOBlock import FNOBlock

class FNO(nn.Module):
    def __init__(self, input_size, output_size, n_dims, num_layers, width, padding=8):
        super(FNO, self).__init__()
        self.n_dims = n_dims
        self.padding = padding
        self.T = num_layers

        self.shallow = nn.Linear(input_size, width)
        self.FNOLayers = nn.ModuleList([FNOBlock(width, modes, n_dims) for _ in range(num_layers)])
        self.projection = nn.Linear(width, output_size)

    def forward(self, x):
        v = nn.Sigmoid(self.shallow(x))
        for t in range(self.T):
            v = self.FNOLayers[t](v)
        return x