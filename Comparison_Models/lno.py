import torch
import torch.nn as nn

class LNOBlock(nn.Module):
    r"""
    Low-rank Neural Operator Block.
    Approximates the integral kernel as \kappa(x, y) = \sum_{j=1}^r \phi_j(x)\psi_j(y).
    This avoids the O(N^2) complexity of dense GNO by factoring the operation,
    similar to unstacked Deep-ONet.
    """
    def __init__(self, channels, rank=64):
        super(LNOBlock, self).__init__()
        self.rank = rank
        
        self.W = nn.Linear(channels, channels)
        
        # Phi and Psi generators map the features from the input space to the rank 'r'
        self.phi_net = nn.Linear(channels, rank)
        
        # Psi network produces a weight for each channel per rank
        # So we use an output of (rank * channels) to allow rich mixing
        self.psi_net = nn.Linear(channels, rank * channels)
        
    def forward(self, x):
        # x shape: (batch, spatial_pts, channels)
        batch, pts, c = x.shape
        
        # phi shape: (batch, spatial_pts, rank)
        phi = self.phi_net(x)
        
        # psi shape: (batch, spatial_pts, rank * c) -> (batch, spatial_pts, rank, c)
        psi = self.psi_net(x).view(batch, pts, self.rank, c)
        
        # Compute the "branch" integral: \int \psi(y) * v(y) dy
        # Element-wise product of psi along channels, then sum over spatial domain
        # x reshaped: (batch, pts, 1, c)
        # c_j = \sum_y \psi_j(y) * v(y)
        # Integral shape: (batch, rank, c)
        integral_c = torch.sum(psi * x.unsqueeze(2), dim=1) / pts # Average over domain
        
        # Recombine with phi(x): \sum_j \phi_j(x) * c_j
        # phi is (batch, pts, rank, 1) to broadcast with (batch, 1, rank, c)
        # Result shape: (batch, pts, c)
        kernel_out = torch.sum(phi.unsqueeze(3) * integral_c.unsqueeze(1), dim=2)
        
        w_x = self.W(x)
        return torch.nn.functional.gelu(w_x + kernel_out)

class LNO(nn.Module):
    """
    Low-rank Neural Operator.
    Constructs a full architecture using multiple LNO layers.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, rank=64, num_layers=4):
        super(LNO, self).__init__()
        
        self.lift = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([LNOBlock(hidden_channels, rank=rank) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x):
        # x shape: (batch, ..., in_channels)
        initial_shape = x.shape[:-1]
        batch_size = x.size(0)
        
        x = x.view(batch_size, -1, x.size(-1))
        
        v = self.lift(x)
        for block in self.blocks:
            v = block(v)
            
        out = self.proj(v)
        
        return out.view(*initial_shape, out.size(-1))
