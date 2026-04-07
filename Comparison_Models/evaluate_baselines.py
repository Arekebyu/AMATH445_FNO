import sys
import os
import argparse
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "Network"))
sys.path.append(os.path.join(base_dir, "Solvers"))

from FNO import FNO
from Comparison_Models import NN, FCN, RBM, PCANN, GNO, MGNO, LNO
from Solvers.generate_data import generate_burgers_data, generate_darcy_data, generate_navier_stokes_data

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate FNO Baselines across Resolutions and PDEs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train each model")
    parser.add_argument("--samples", type=int, default=500, help="Number of data samples to generate")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--runs", type=int, default=1, help="Number of random seeds to run")
    return parser.parse_args()

def prepare_data(pde, resolution, samples):
    if pde == "burgers":
        X, Y = generate_burgers_data(n_samples=samples, nx=resolution)
        x_grid = np.linspace(0, 1, resolution)
        x_g = np.repeat(x_grid[None, :, None], samples, axis=0)
        X = X[..., None]
        X_full = np.concatenate([X, x_g], axis=-1)
        Y = Y[..., None]
        n_dims = 1
        in_channels = 2
        out_channels = 1
        return X_full, Y, n_dims, in_channels, out_channels
        
    elif pde == "darcy":
        # Darcy solver generates grid nx+2 but solver config takes nx
        # Here resolution specifies the target nx grid
        X, Y = generate_darcy_data(n_samples=samples, nx=resolution)
        nx_full = X.shape[1] # nx + 2 effectively
        x_grid = np.linspace(0, 1, nx_full)
        y_grid = np.linspace(0, 1, nx_full)
        X_G, Y_G = np.meshgrid(x_grid, y_grid, indexing='ij')
        x_g = np.repeat(X_G[None, ..., None], samples, axis=0)
        y_g = np.repeat(Y_G[None, ..., None], samples, axis=0)
        X = X[..., None]
        X_full = np.concatenate([X, x_g, y_g], axis=-1)
        Y = Y[..., None]
        n_dims = 2
        in_channels = 3
        out_channels = 1
        return X_full, Y, n_dims, in_channels, out_channels
        
    elif pde == "navier_stokes":
        X, Y = generate_navier_stokes_data(n_samples=samples, nx=resolution)
        nx_full = X.shape[1]
        x_grid = np.linspace(0, 1, nx_full, endpoint=False)
        y_grid = np.linspace(0, 1, nx_full, endpoint=False)
        X_G, Y_G = np.meshgrid(x_grid, y_grid, indexing='ij')
        x_g = np.repeat(X_G[None, ..., None], samples, axis=0)
        y_g = np.repeat(Y_G[None, ..., None], samples, axis=0)
        X = X[..., None]
        X_full = np.concatenate([X, x_g, y_g], axis=-1)
        Y = Y[..., None]
        n_dims = 2
        in_channels = 3
        out_channels = 1
        return X_full, Y, n_dims, in_channels, out_channels
    else:
        raise ValueError(f"Unknown PDE: {pde}")

def build_model(model_name, n_dims, in_channels, out_channels):
    # Base hyperparams configured to fit reasonably on a standard GPU
    modes = 12 if n_dims == 2 else 16
    width = 32 if n_dims == 2 else 64
    hidden = 32
    
    if model_name == "FNO":
        num_modes = (modes, modes) if n_dims == 2 else (modes,)
        return FNO(modes=num_modes, input_size=in_channels, output_size=out_channels, n_dims=n_dims, num_layers=4, width=width)
    elif model_name == "NN":
        return NN(in_channels, out_channels, hidden_channels=hidden)
    elif model_name == "FCN":
        return FCN(in_channels, out_channels, n_dims=n_dims, hidden_channels=hidden)
    elif model_name == "RBM":
        return RBM(in_channels, out_channels, n_modes=16)
    elif model_name == "PCANN":
        return PCANN(in_channels, out_channels, n_modes=16, hidden_channels=hidden)
    elif model_name == "GNO":
        return GNO(in_channels, out_channels, hidden_channels=hidden)
    elif model_name == "MGNO":
        return MGNO(in_channels, out_channels, hidden_channels=hidden, n_dims=n_dims)
    elif model_name == "LNO":
        return LNO(in_channels, out_channels, hidden_channels=hidden, rank=16)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_and_eval(model, train_loader, test_loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    for ep in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            optimizer.step()
            
    train_time = time.time() - start_time
    
    # Eval
    model.eval()
    test_err = 0.0
    start_inf = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            err = torch.norm(out.reshape(out.shape[0], -1) - y_batch.reshape(y_batch.shape[0], -1), 2, dim=1) / \
                  (torch.norm(y_batch.reshape(y_batch.shape[0], -1), 2, dim=1) + 1e-8)
            test_err += err.sum().item()
            
    inf_time = time.time() - start_inf
    test_err /= len(test_loader.dataset)
    
    return train_time, inf_time, test_err

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pde_configs = {
        "burgers": [256, 512, 1024],
        "darcy": [8, 16, 32, 64],
        "navier_stokes": [8, 16, 32, 64]
    }
    
    models_to_test = ["FNO", "NN", "FCN", "RBM", "PCANN", "GNO", "MGNO", "LNO"]
    results = []
    
    for pde, resolutions in pde_configs.items():
        print(f"\n========================================")
        print(f"   Evaluating PDE: {pde.upper()}")
        print(f"========================================")
        
        for res in resolutions:
            print(f"\n[ Resolution: {res} ]")
            X_full, Y, n_dims, in_channels, out_channels = prepare_data(pde, res, args.samples)
            
            n_train = int(0.8 * len(X_full))
            X_train_tensor = torch.tensor(X_full[:n_train], dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y[:n_train], dtype=torch.float32)
            X_test_tensor = torch.tensor(X_full[n_train:], dtype=torch.float32)
            Y_test_tensor = torch.tensor(Y[n_train:], dtype=torch.float32)
            
            train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=args.batch_size, shuffle=False)
            
            for model_name in models_to_test:
                print(f"  -> Model: {model_name}...")
                model = build_model(model_name, n_dims, in_channels, out_channels).to(device)
                
                if hasattr(model, 'fit_pca'):
                    model.fit_pca(X_train_tensor.to(device), Y_train_tensor.to(device))
                
                try:
                    train_time, inf_time, test_err = train_and_eval(model, train_loader, test_loader, args.epochs, device)
                    print(f"     Rel L2 Error: {test_err:.4f}  |  Train Time: {train_time:.2f}s  |  Inf Time: {inf_time:.2f}s")
                    results.append({
                        "PDE": pde,
                        "Resolution": res,
                        "Model": model_name,
                        "Rel_L2_Error": test_err,
                        "Train_Time_s": train_time,
                        "Inference_Time_s": inf_time
                    })
                except Exception as e:
                    print(f"     [!] Failed to train {model_name}: {e}")
                    results.append({
                        "PDE": pde,
                        "Resolution": res,
                        "Model": model_name,
                        "Rel_L2_Error": np.nan,
                        "Train_Time_s": np.nan,
                        "Inference_Time_s": np.nan
                    })
    
    # Export results
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\n[Done] Benchmark complete! Results saved to 'benchmark_results.csv'.")

if __name__ == "__main__":
    main()
