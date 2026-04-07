import torch
import torch.nn as nn

class PCANN(nn.Module):
    """
    PCA Autoencoder + Neural Network (PCANN).
    Uses PCA as an autoencoder on both input and output data, and
    interpolates the latent spaces with a neural network.
    """
    def __init__(self, in_channels, out_channels, n_modes=16, hidden_channels=64, num_layers=4):
        super(PCANN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        
        # PCA projection matrices
        self.register_buffer('V_in', None)   # Principal components for input
        self.register_buffer('mean_in', None)
        
        self.register_buffer('V_out', None)  # Principal components for output
        self.register_buffer('mean_out', None)
        
        # Latent network
        layers = []
        layers.append(nn.Linear(n_modes * in_channels, hidden_channels))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, n_modes * out_channels))
        
        self.latent_net = nn.Sequential(*layers)
        self.pca_fitted = False
        
    def fit_pca(self, X, Y=None):
        """
        Fits the PCA bases explicitly before training the latent network.
        X, Y are expected to be of shape (batch, grid_x, [grid_y], ..., channels)
        The grid dimensions are flattened out to extract spatial covariance.
        """
        # Flatten spatial dimensions: (batch, spatial_pts, channels)
        batch = X.shape[0]
        x_flat = X.view(batch, -1, self.in_channels)
        x_pts = x_flat.shape[1]
        
        # Treat batch dimensions as independent samples for the mode decomposition over space
        # Covariance matrix size: (spatial_pts, spatial_pts) per channel, or global flat
        # For standard PCANN, modes are spatial.
        # X: (batch, spatial_pts * channels)
        x_flat = X.reshape(batch, -1)
        mean_in = torch.mean(x_flat, dim=0, keepdim=True)
        x_centered = x_flat - mean_in
        
        # Compute SVD
        U, S, V = torch.pca_lowrank(x_centered, q=self.n_modes * self.in_channels, center=False)
        self.V_in = V # shape: (features, n_modes * in_channels)
        self.mean_in = mean_in
        
        if Y is not None:
            y_flat = Y.reshape(batch, -1)
            mean_out = torch.mean(y_flat, dim=0, keepdim=True)
            y_centered = y_flat - mean_out
            U_y, S_y, V_y = torch.pca_lowrank(y_centered, q=self.n_modes * self.out_channels, center=False)
            self.V_out = V_y
            self.mean_out = mean_out
            
        self.pca_fitted = True
        
    def forward(self, x):
        if not self.pca_fitted or self.V_in is None:
            raise RuntimeError("PCA bases have not been fit. Call fit_pca(X, Y) with training subset prior to forward passes.")
            
        initial_shape = x.shape[:-1]
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        
        # Project to latent space
        z_in = torch.matmul(x_flat - self.mean_in, self.V_in)
        
        # Neural Network mapping
        z_out = self.latent_net(z_in)
        
        # Project back to output space
        # Check if output PCA is fit, else just assume it maps to input space or something similar (predicting single steps)
        if self.V_out is not None:
            y_flat = torch.matmul(z_out, self.V_out.T) + self.mean_out
        else:
            # If no V_out is fitted, this means autoencoding the same state directly or projecting symmetrically.
            # Usually V_out is fit if mapping between distinct fields.
            y_flat = torch.matmul(z_out, self.V_in.T) + self.mean_in
            
        # Reshape to expected spatial resolution
        # For simplicity, returning the flattened out states because we assume symmetric grids, 
        # but to keep the input shape we reshape properly assuming y_flat matches out_channels.
        # Compute spatial points from initial shape
        spatial_pts = int(y_flat.shape[-1] / self.out_channels)
        return y_flat.view(batch, -1, self.out_channels).view(*initial_shape, self.out_channels)
