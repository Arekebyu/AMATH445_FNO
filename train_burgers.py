import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "Network"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Solvers"))
from FNO import FNO
from generate_data import generate_burgers_data

def generate_data(samples=500, location="data/burgers_data.npz"):
    nx = 64
    X, Y = generate_burgers_data(n_samples=samples, nx=nx)
    
    # Save data to data/burgers_data.npz
    np.savez(location, X=X, Y=Y)

def train_burgers(data_path, proportion=0.8, epochs=50, modes=16, width=64):
    print("========================================")
    print("    Training FNO on Burgers' Equation   ")
    print("========================================")

    # 1. Generate Data
    data = np.load(data_path)
    X, Y = data['X'], data['Y']
    n_train = int(proportion * len(X))
    n_test = len(X) - n_train
    
    # We append exact grid locations as an input channel a(x) -> (a(x), x)
    x_grid = np.linspace(0, 1, X.shape[1])
    x_grid = np.repeat(x_grid[None, :, None], X.shape[0], axis=0)
    
    X = X[..., None]
    X_full = np.concatenate([X, x_grid], axis=-1) # shape: (samples, nx, 2)
    Y = Y[..., None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, Y_train = torch.tensor(X_full[:n_train], dtype=torch.float32), torch.tensor(Y[:n_train], dtype=torch.float32)
    X_test, Y_test = torch.tensor(X_full[-n_test:], dtype=torch.float32), torch.tensor(Y[-n_test:], dtype=torch.float32)

    batch_size = min(20, n_train)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    # 2. Setup Model
    model = FNO(modes=(modes,), input_size=2, output_size=1, n_dims=1, num_layers=4, width=width).to(device)
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
                err = torch.norm(out.view(out.shape[0], -1) - y_batch.view(y_batch.shape[0], -1), 2, dim=1) / \
                      (torch.norm(y_batch.view(y_batch.shape[0], -1), 2, dim=1) + 1e-8)
                test_err += err.sum().item()

        test_err /= len(X_test)
        
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1:03d}/{epochs} | Train Loss: {train_loss/len(train_loader):.5f} | Test Rel L2: {test_err:.5f}")
    
    torch.save(model.state_dict(), "burgers_fno.pth")
    print("Model saved to burgers_fno.pth")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/burgers_data_500.npz"):
        generate_data(samples=500, location="data/burgers_data_500.npz")
    train_burgers(data_path="data/burgers_data_500.npz")
