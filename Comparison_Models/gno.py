import torch
import torch.nn as nn

class DenseGNOBlock(nn.Module):
    """
    A single step of Graph Neural Operator on a dense grid using an attention-like 
    inner-product mechanism to evaluate the integral kernel over the full spatial domain.
    Integral kernel acts as fully-connected message passing.
    """
    def __init__(self, channels):
        super(DenseGNOBlock, self).__init__()
        self.W = nn.Linear(channels, channels)
        
        # Kernel networks for pairwise message evaluation
        # For simplicity and tractability on dense grids, we use a factorized kernel (Attention)
        self.K_query = nn.Linear(channels, channels)
        self.K_key   = nn.Linear(channels, channels)
        self.K_value = nn.Linear(channels, channels)
        
    def forward(self, x):
        # x shape: (batch, spatial_pts, channels)
        
        # Integral transformation
        q = self.K_query(x)
        k = self.K_key(x)
        v = self.K_value(x)
        
        # Compute kernel values K(x, y) via inner product
        # Shape: (batch, spatial_pts, spatial_pts)
        kernel = torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        
        # The integral is approximated as a sum over the grid
        integral = torch.bmm(kernel, v) / x.size(1) # normalize by grid size
        
        # Local transformation
        w_x = self.W(x)
        
        return torch.nn.functional.gelu(w_x + integral)

class GNO(nn.Module):
    """
    Graph Neural Operator (Dense Grid Implementation).
    Uses a standard lifting/projection layer with multiple dense integral kernel blocks.
    To avoid O(N^2) explosion, this implementation assumes standard grids are relatively small
    or uses random spatial permutations.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4):
        super(GNO, self).__init__()
        
        self.lift = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([DenseGNOBlock(hidden_channels) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x):
        # x shape: (batch, ..., in_channels)
        initial_shape = x.shape[:-1]
        batch_size = x.size(0)
        
        # Flatten spatial dimensions for dense graph formulation
        x = x.view(batch_size, -1, x.size(-1))
        
        v = self.lift(x)
        for block in self.blocks:
            v = block(v)
            
        out = self.proj(v)
        
        # Reshape to original spatial format
        return out.view(*initial_shape, out.size(-1))
