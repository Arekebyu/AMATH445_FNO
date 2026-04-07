import torch
import torch.nn as nn
import torch.nn.functional as F
from .gno import DenseGNOBlock

class MGNO(nn.Module):
    """
    Multipole Graph Neural Operator (Dense Grid Implementation).
    Operates the kernel integrations at multiple spatial scales using average pooling 
    to prevent dense O(N^2) bottlenecks on large grids.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4, n_dims=1):
        super(MGNO, self).__init__()
        
        self.lift = nn.Linear(in_channels, hidden_channels)
        self.proj = nn.Linear(hidden_channels, out_channels)
        self.n_dims = n_dims
        
        self.blocks = nn.ModuleList([DenseGNOBlock(hidden_channels) for _ in range(num_layers)])
        
    def forward(self, x):
        # x shape: (batch, grid..., channels)
        batch_size = x.size(0)
        initial_shape = x.shape[:-1]
        
        v = self.lift(x)
        
        # To apply MGNO, we do pooling prior to each consecutive dense GNO block to evaluate coarsely
        # For generalization across dimensions without explicit 1D/2D pooling, we will use
        # adaptive pooling based on the grid flattening or rely on simple channel pooling
        # To make it robust for arbitrary grids, we pool the flattened spatial dimension:
        # v shape: (batch, spatial_pts, hidden)
        
        v_flat = v.view(batch_size, -1, v.size(-1))
        current_pts = v_flat.size(1)
        
        for i, block in enumerate(self.blocks):
            # Downsample the spatial sequence simply by stride
            # E.g. progressively halved resolutions: current_pts // (2 ** i)
            scale_factor = max(1, 2 ** i)
            pooled_pts = max(1, current_pts // scale_factor)
            
            # Simple average pooling over flattened domain
            v_pool = F.adaptive_avg_pool1d(v_flat.transpose(1, 2), pooled_pts).transpose(1, 2)
            
            # Apply GNO block on pooled space
            v_pool = block(v_pool)
            
            # Interpolate back
            v_up = F.interpolate(v_pool.transpose(1, 2), size=current_pts, mode='linear', align_corners=False).transpose(1, 2)
            v_flat = v_flat + v_up # Residual connection across scales
            
        out = self.proj(v_flat)
        return out.view(*initial_shape, -1)
