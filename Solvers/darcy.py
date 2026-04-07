import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

class Darcy2D:
    """
    2D Darcy Flow Solver: -div(a(x) grad u(x)) = f(x)
    Uses 5-point finite difference stencil.
    Dirichlet boundary conditions: u=0.
    """
    def __init__(self, nx=64, L=1.0):
        self.nx = nx
        self.L = L
        self.dx = L / (nx + 1)
        self.x = np.linspace(0, L, nx + 2)[1:-1]
        self.y = np.linspace(0, L, nx + 2)[1:-1]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.N = nx * nx

    def solve(self, a, f=None):
        """
        Solve for u given permeability a(x,y).
        a: (nx+2, nx+2) array
        f: (nx, nx) array (default 1.0)
        """
        nx = self.nx
        h2 = self.dx**2
        
        if f is None:
            f = np.ones((nx, nx))
            
        # Staggered coefficients
        # a_mid_x(i, j) = 0.5 * (a(i, j) + a(i+1, j))
        # a_mid_y(i, j) = 0.5 * (a(i, j) + a(i, j+1))
        
        # We need the values of a at the boundaries too
        a_mid_x = 0.5 * (a[1:-1, :-1] + a[1:-1, 1:])
        a_mid_y = 0.5 * (a[:-1, 1:-1] + a[1:, 1:-1])
        
        # Build sparse matrix A
        # Internal nodes index: k = j * nx + i
        data = []
        rows = []
        cols = []
        
        for j in range(nx):
            for i in range(nx):
                k = j * nx + i
                
                # Center coefficient
                center = (a_mid_x[j, i+1] + a_mid_x[j, i] + 
                          a_mid_y[j+1, i] + a_mid_y[j, i]) / h2
                data.append(center)
                rows.append(k)
                cols.append(k)
                
                # Left
                if i > 0:
                    data.append(-a_mid_x[j, i] / h2)
                    rows.append(k)
                    cols.append(k - 1)
                
                # Right
                if i < nx - 1:
                    data.append(-a_mid_x[j, i+1] / h2)
                    rows.append(k)
                    cols.append(k + 1)
                    
                # Bottom
                if j > 0:
                    data.append(-a_mid_y[j, i] / h2)
                    rows.append(k)
                    cols.append(k - nx)
                    
                # Top
                if j < nx - 1:
                    data.append(-a_mid_y[j+1, i] / h2)
                    rows.append(k)
                    cols.append(k + nx)
                    
        A = sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        b = f.flatten()
        
        u_flat = splinalg.spsolve(A, b)
        u = u_flat.reshape((nx, nx))
        
        # Pad with 0 as Dirichlet boundary
        u_full = np.zeros((nx+2, nx+2))
        u_full[1:-1, 1:-1] = u
        
        return u_full

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    nx = 64
    solver = Darcy2D(nx=nx)
    
    # Random permeability a
    # a must be strictly positive
    a = np.exp(np.random.normal(size=(nx+2, nx+2)) * 0.5)
    
    u = solver.solve(a)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    # Corrected indexing for permeability visualization
    plt.imshow(a, extent=[0, 1, 0, 1])
    plt.title("Permeability a(x,y)")
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(u, extent=[0, 1, 0, 1])
    plt.title("Pressure u(x,y)")
    plt.colorbar()
    plt.show()
