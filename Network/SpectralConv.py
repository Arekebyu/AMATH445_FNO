import torch
import torch.nn as nn

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, n_dims):
        """
        FFT -> Filter -> Linear transform -> Inverse FFT
        """
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)

        self.weights = nn.ParameterList([
            nn.parameter(
                self.scale * torch.rand(in_channels, out_channels, *modes, dtype=troch.cfloat)
            )for _ in range(2**(n_dims - 1))]
        )        

    def forward(self, x):
        # x: (batch, channels, dim1, dim2, ... dim_n)
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=list(range(-self.n_dims, 0)))
        out_ft = torch.zeros(batchsize, self.out_channels, x.shape[2], x.shape[3]//2+1, dtype=torch.cfloat, device=x.device)

        # convolution is multiplication in frequency domain
        for i, combo in enumerate(product([True, False], repeat=self.n_dims-1)):
            # Build the slice for the input/output frequency tensor
            # Indexing: Batch, Channel, Dim1, Dim2...
            slices = [slice(None), slice(None)] 
            
            for d in range(self.n_dims - 1):
                if combo[d]: # Start of the dimension
                    slices.append(slice(0, self.modes[d]))
                else:        # End of the dimension
                    slices.append(slice(-self.modes[d], None))
            
            # Last dimension is always the first 'k' modes due to rfft
            slices.append(slice(0, self.modes[-1]))

            # Apply complex multiplication for this corner
            out_ft[slices] = torch.einsum("bixy,ioxy->boxy", x_ft[slices], self.weights[i])

        # convert back to time domain
        x = torch.fft.irfftn(out_ft, s=x.shape[-self.n_dims:])
        return x
