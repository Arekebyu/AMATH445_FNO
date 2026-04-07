import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "Network"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Solvers"))
from FNO import FNO
from generate_data import generate_navier_stokes_data

def generate_data(samples=500, location="data/navier_stokes_data.npz"):
    nx = 64
    X, Y = generate_navier_stokes_data(n_samples=samples, nx=nx)
    
    # Save data to data/navier_stokes_data.npz
    np.savez(location, X=X, Y=Y)
    

def train_navier_stokes(data_path, proportion=0.8, epochs=100, modes=12, width=32):
    print("========================================")
    print("  Training FNO on Navier-Stokes 2D      ")
    print("========================================")
    
    # Load data
    data = np.load(data_path)
    X, Y = data['X'], data['Y']

    nx = 64
    n_train = int(proportion * len(X))
    n_test = len(X) - n_train
    
    # Grid locations (x, y)
    x_grid = np.linspace(0, 1, nx, endpoint=False)
    y_grid = np.linspace(0, 1, nx, endpoint=False)
    X_G, Y_G = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    x_g = np.repeat(X_G[None, ..., None], X.shape[0], axis=0)
    y_g = np.repeat(Y_G[None, ..., None], X.shape[0], axis=0)
    
    X = X[..., None]
    X_full = np.concatenate([X, x_g, y_g], axis=-1) # shape: (samples, nx, nx, 3)
    Y = Y[..., None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, Y_train = torch.tensor(X_full[:n_train], dtype=torch.float32), torch.tensor(Y[:n_train], dtype=torch.float32)
    X_test, Y_test = torch.tensor(X_full[-n_test:], dtype=torch.float32), torch.tensor(Y[-n_test:], dtype=torch.float32)

    batch_size = min(20, n_train)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    # 2. Setup Model
    # input_size=3 (w0, x, y)
    model = FNO(modes=(modes, modes), input_size=3, output_size=1, n_dims=2, num_layers=4, width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()

    # 3. Train
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Relative L2 Error
        model.eval()
        test_err = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                err = torch.norm(out.reshape(out.shape[0], -1) - y_batch.reshape(y_batch.shape[0], -1), 2, dim=1) / \
                      (torch.norm(y_batch.reshape(y_batch.shape[0], -1), 2, dim=1) + 1e-8)
                test_err += err.sum().item()

        test_err /= len(X_test)
        
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1:03d}/{epochs} | Train Loss: {train_loss/len(train_loader):.5f} | Test Rel L2: {test_err:.5f}")
    
    # Save model
    torch.save(model.state_dict(), "navier_stokes_fno.pth")
    print("Model saved to navier_stokes_fno.pth")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/navier_stokes_data_500.npz"):
        generate_data(samples=500, location="data/navier_stokes_data_500.npz")
    train_navier_stokes(data_path="data/navier_stokes_data_500.npz")
