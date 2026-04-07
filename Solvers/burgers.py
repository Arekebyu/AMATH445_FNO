import numpy as np
from scipy.fft import fft, ifft

class Burgers1D:
    """
    1D Burgers' Equation Solver: u_t + u*u_x = nu*u_xx
    Uses a pseudo-spectral method in space and RK4 in time.
    """
    def __init__(self, L=1.0, nx=256, nu=0.01):
        self.L = L
        self.nx = nx
        self.nu = nu
        self.dx = L / nx
        self.x = np.linspace(0, L, nx, endpoint=False)
        
        # Wavenumbers for spectral derivatives
        self.k = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        
    def solve(self, u0, T, dt=1e-3):
        """
        Solve the equation from t=0 to t=T with time step dt.
        u0: Initial condition (spatial domain)
        """
        # CFL Condition for convective term
        max_u = np.max(np.abs(u0))
        if max_u > 0:
            dt_cfl = 0.5 * self.dx / max_u
            dt = min(dt, dt_cfl)
            
        nt = int(np.ceil(T / dt))
        dt = T / nt
        
        u_hat = fft(u0)
        
        # Integrating factor exponentials
        E = np.exp(-self.nu * (self.k**2) * dt)
        E2 = np.exp(-self.nu * (self.k**2) * dt / 2.0)
        
        # Helper for non-linear term with 2/3 de-aliasing
        k_max = np.max(self.k)
        dealias_mask = np.abs(self.k) < (2.0 / 3.0) * k_max
        
        def nonlin(u_h):
            u_h_clean = u_h * dealias_mask
            u = np.real(ifft(u_h_clean))
            u_sq_hat = fft(u**2)
            return -0.5j * self.k * u_sq_hat
        
        # IF-RK4 Time Stepping
        for _ in range(nt):
            k1 = nonlin(u_hat)
            k2 = nonlin(E2 * (u_hat + 0.5 * dt * k1))
            k3 = nonlin(E2 * u_hat + 0.5 * dt * k2)
            k4 = nonlin(E * u_hat + dt * E2 * k3)
            
            u_hat = E * u_hat + (dt / 6.0) * (E * k1 + 2 * E2 * (k2 + k3) + k4)
            
        return np.real(ifft(u_hat))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Parameters
    nx = 512
    nu = 0.01
    T = 1.0
    dt = 0.001
    
    solver = Burgers1D(nx=nx, nu=nu)
    
    # Initial condition: u(x,0) = sin(2*pi*x)
    u0 = np.sin(2 * np.pi * solver.x)
    
    u_final = solver.solve(u0, T, dt)
    
    plt.plot(solver.x, u0, label='Initial')
    plt.plot(solver.x, u_final, label=f'Final (T={T})')
    plt.legend()
    plt.title("1D Burgers' Equation (Pseudo-spectral RK4)")
    plt.show()
