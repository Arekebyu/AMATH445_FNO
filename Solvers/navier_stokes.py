import numpy as np
from scipy.fft import fft2, ifft2

class NavierStokes2D:
    """
    2D Navier-Stokes Solver (Vorticity formulation)
    w_t + u.grad(w) = nu * div^2(w) + f
    Uses pseudo-spectral method with periodic BC.
    """
    def __init__(self, nx=64, ny=64, L=1.0, nu=1e-3):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.nu = nu
        
        self.dx = L / nx
        self.dy = L / ny
        
        # Wavenumbers
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')
        
        # Laplacian operator in Fourier space
        self.Lap = -(self.Kx**2 + self.Ky**2)
        
        # Inverse Laplacian for solving Poisson eq (u, v from w)
        # Avoid division by zero at k=0
        self.InvLap = np.zeros_like(self.Lap)
        self.InvLap[self.Lap != 0] = 1.0 / self.Lap[self.Lap != 0]

    def _get_velocity(self, w_hat):
        """
        Compute velocity (u, v) from vorticity w in Fourier space.
        w = v_x - u_y = i*kx*v_hat - i*ky*u_hat
        Streamfunction psi: Lap(psi) = -w
        u = psi_y, v = -psi_x
        """
        psi_hat = -self.InvLap * w_hat
        u_hat = 1j * self.Ky * psi_hat
        v_hat = -1j * self.Kx * psi_hat
        
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        return u, v

    def solve(self, w0, f, T, dt=1e-3):
        """
        w0: Initial vorticity (spatial)
        f: Forcing term (spatial)
        """
        w_hat_0 = fft2(w0)
        u0_field, v0_field = self._get_velocity(w_hat_0)
        
        # CFL Condition for convective term
        max_v = max(np.max(np.abs(u0_field)), np.max(np.abs(v0_field)))
        if max_v > 0:
            dt_cfl = 0.5 * min(self.dx, self.dy) / max_v
            dt = min(dt, dt_cfl)
            
        nt = int(np.ceil(T / dt))
        dt = T / nt
        
        w_hat = w_hat_0.copy()
        f_hat = fft2(f)
        
        # Integrating factor exponentials
        E = np.exp(self.nu * self.Lap * dt)
        E2 = np.exp(self.nu * self.Lap * dt / 2.0)
        
        # 2D 2/3 De-aliasing mask
        kmax_x, kmax_y = np.max(self.Kx), np.max(self.Ky)
        dealias_mask = (np.abs(self.Kx) < (2.0/3.0)*kmax_x) & (np.abs(self.Ky) < (2.0/3.0)*kmax_y)
        
        def nonlin(w_h):
            w_h_clean = w_h * dealias_mask
            u, v = self._get_velocity(w_h_clean)
            
            w_x_hat = 1j * self.Kx * w_h_clean
            w_y_hat = 1j * self.Ky * w_h_clean
            
            w_x = np.real(ifft2(w_x_hat))
            w_y = np.real(ifft2(w_y_hat))
            
            advection = -(u * w_x + v * w_y)
            return fft2(advection) + f_hat
            
        # IF-RK4 Time Stepping
        for _ in range(nt):
            k1 = nonlin(w_hat)
            k2 = nonlin(E2 * (w_hat + 0.5 * dt * k1))
            k3 = nonlin(E2 * w_hat + 0.5 * dt * k2)
            k4 = nonlin(E * w_hat + dt * E2 * k3)
            
            w_hat = E * w_hat + (dt / 6.0) * (E * k1 + 2 * E2 * (k2 + k3) + k4)
            
        return np.real(ifft2(w_hat))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    nx, ny = 64, 64
    solver = NavierStokes2D(nx, ny, nu=1e-3)
    
    # Grid
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial Condition: Random small perturbations
    w0 = np.random.normal(scale=0.1, size=(nx, ny))
    
    # Forcing: sin(2*pi*(x+y)) + cos(2*pi*(x+y))
    f = 0.1 * (np.sin(2*np.pi*(X+Y)) + np.cos(2*np.pi*(X+Y)))
    
    w_final = solver.solve(w0, f, T=1.0, dt=1e-3)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(w0.T, origin='lower', extent=[0, 1, 0, 1])
    plt.title("Initial Vorticity")
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(w_final.T, origin='lower', extent=[0, 1, 0, 1])
    plt.title("Final Vorticity (T=1.0)")
    plt.colorbar()
    plt.show()
