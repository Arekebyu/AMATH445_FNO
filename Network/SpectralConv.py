import torch
import torch.nn as nn
from itertools import product

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, n_dims):
        """
        FFT -> Filter -> Linear transform -> Inverse FFT
        """
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dims = n_dims

        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)

        self.weights = nn.ParameterList([
            nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, *modes, dtype=torch.cfloat)
            ) for _ in range(2**(n_dims - 1))]
        )        

    def forward(self, x):
        # x: (batch, channels, dim1, dim2, ... dim_n)
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=list(range(-self.n_dims, 0)))
        
        # Initialize output shape dynamically
        out_ft_shape = list(x.shape)
        out_ft_shape[1] = self.out_channels
        out_ft_shape[-1] = out_ft_shape[-1] // 2 + 1
        
        out_ft = torch.zeros(*out_ft_shape, dtype=torch.cfloat, device=x.device)

        # Cap the modes dynamically to handle evaluating on sparse grids smaller than the modes parameterized
        actual_modes = []
        for d in range(self.n_dims):
            n = x_ft.shape[d + 2]
            if d < self.n_dims - 1:
                actual_modes.append(min(self.modes[d], n // 2))
            else:
                actual_modes.append(min(self.modes[d], n))

        # convolution is multiplication in frequency domain
        for i, combo in enumerate(product([True, False], repeat=self.n_dims-1)):
            slices = [slice(None), slice(None)] 
            weight_slices = [slice(None), slice(None)]
            
            for d in range(self.n_dims - 1):
                if combo[d]: # Start of the dimension
                    slices.append(slice(0, actual_modes[d]))
                else:        # End of the dimension
                    slices.append(slice(-actual_modes[d], None))
                weight_slices.append(slice(0, actual_modes[d]))
            
            # Last dimension is always the first 'k' modes due to rfft
            slices.append(slice(0, actual_modes[-1]))
            weight_slices.append(slice(0, actual_modes[-1]))

            # Subset the global weights if testing on grids with extremely low resolution
            weights_subset = self.weights[i][tuple(weight_slices)]

            # Apply complex multiplication for this corner
            if self.n_dims == 1:
                out_ft[tuple(slices)] = torch.einsum("bix,iox->box", x_ft[tuple(slices)], weights_subset)
            elif self.n_dims == 2:
                out_ft[tuple(slices)] = torch.einsum("bixy,ioxy->boxy", x_ft[tuple(slices)], weights_subset)
            elif self.n_dims == 3:
                out_ft[tuple(slices)] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[tuple(slices)], weights_subset)
            else:
                raise ValueError(f"n_dims {self.n_dims} not supported")

        # convert back to time domain
        x = torch.fft.irfftn(out_ft, s=x.shape[-self.n_dims:])
        return x
