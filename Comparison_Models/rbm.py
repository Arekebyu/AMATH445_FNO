import torch
import torch.nn as nn
from .pcann import PCANN

class RBM(PCANN):
    """
    Classical Reduced Basis Method (RBM) utilizing a POD basis approach.
    Similar to PCANN, but the mapping in the latent space is strictly a linear operation.
    """
    def __init__(self, in_channels, out_channels, n_modes=16):
        # We inherit PCANN, but override the latent net to be a simple Linear layer without activations
        super(RBM, self).__init__(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  n_modes=n_modes, 
                                  hidden_channels=1, # unused
                                  num_layers=1)      # unused
        
        # Override with a simple linear regression mapping
        self.latent_net = nn.Linear(n_modes * in_channels, n_modes * out_channels)
