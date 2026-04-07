import numpy as np
from burgers import Burgers1D
from darcy import Darcy2D
from navier_stokes import NavierStokes2D

class GRF:
    """
    Gaussian Random Field Sampler (Periodic)
    """
    def __init__(self, n, L=1.0, alpha=2.0, tau=3.0, dims=1):
        self.n = n
        self.L = L
        self.alpha = alpha
        self.tau = tau
        self.dims = dims
        
        if dims == 1:
            k = np.fft.fftfreq(n) * n
            self.K = k**2
        elif dims == 2:
            kx = np.fft.fftfreq(n) * n
            ky = np.fft.fftfreq(n) * n
            KX, KY = np.meshgrid(kx, ky)
            self.K = KX**2 + KY**2

    def sample(self):
        """
        Sample in Fourier space
        """
        noise = np.random.normal(size=self.K.shape) + 1j * np.random.normal(size=self.K.shape)
        
        # Power law filter: (tau^2 + |k|^2)^{-alpha/2}
        # In original FNO, alpha is usually around 2.0-3.0
        # and tau is a length scale parameter.
        filt = (self.tau**2 + self.K)**(-self.alpha / 2.0)
        
        # Applying filter and transforming back
        if self.dims == 1:
            a_hat = noise * filt
            a = np.real(np.fft.ifft(a_hat))
        else:
            a_hat = noise * filt
            a = np.real(np.fft.ifft2(a_hat))
            
        # Normalize to have zero mean and unit variance (optional)
        a = (a - np.mean(a)) / np.std(a)
        return a

def generate_burgers_data(n_samples=10, nx=128):
    print(f"Generating {n_samples} Burgers samples...")
    grf = GRF(n=nx, dims=1, alpha=2.5, tau=5.0)
    solver = Burgers1D(nx=nx, nu=0.01)
    
    inputs = []
    outputs = []
    
    for i in range(n_samples):
        u0 = grf.sample()
        uT = solver.solve(u0, T=1.0, dt=1e-3)
        inputs.append(u0)
        outputs.append(uT)
        
    return np.array(inputs), np.array(outputs)

def generate_darcy_data(n_samples=10, nx=64):
    print(f"Generating {n_samples} Darcy samples...")
    # Darcy permeability needs to be positive
    grf = GRF(n=nx+2, dims=2, alpha=2.0, tau=3.0)
    solver = Darcy2D(nx=nx)
    
    inputs = []
    outputs = []
    
    for i in range(n_samples):
        # Exp(GRF) ensures positivity
        a = np.exp(grf.sample())
        u = solver.solve(a)
        inputs.append(a)
        outputs.append(u)
        
    return np.array(inputs), np.array(outputs)

def generate_navier_stokes_data(n_samples=10, nx=64):
    print(f"Generating {n_samples} Navier-Stokes samples...")
    grf = GRF(n=nx, dims=2, alpha=2.5, tau=2.5)
    solver = NavierStokes2D(nx=nx, ny=nx, nu=1e-3)
    
    inputs = []
    outputs = []
    
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, nx, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = 0.1 * (np.sin(2*np.pi*(X+Y)) + np.cos(2*np.pi*(X+Y)))

    for i in range(n_samples):
        w0 = grf.sample()
        w_final = solver.solve(w0, f, T=1.0, dt=1e-3)
        inputs.append(w0)
        outputs.append(w_final)

    return np.array(inputs), np.array(outputs)

if __name__ == "__main__":
    # Example generation
    X_burgers, Y_burgers = generate_burgers_data(5)
    print(f"Burgers data: {X_burgers.shape}, {Y_burgers.shape}")
    
    X_darcy, Y_darcy = generate_darcy_data(5)
    print(f"Darcy data: {X_darcy.shape}, {Y_darcy.shape}")
