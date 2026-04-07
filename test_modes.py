import sys
import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "Network"))
sys.path.append(os.path.join(base_dir, "Comparison_Models"))

from FNO import FNO
from evaluate_baselines import prepare_data, train_and_eval

def get_args():
    parser = argparse.ArgumentParser(description="Experiment FNO Modes across Resolutions and PDEs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train each model")
    parser.add_argument("--samples", type=int, default=500, help="Number of data samples to generate")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Resolutions to test
    pde_configs = {
        "burgers": [128, 256, 512, 1024],
        "darcy": [8, 16, 32, 64],
        "navier_stokes": [8, 16, 32, 64]
    }
    
    # Mode variations relative to default (0 is default)
    # Default for Burgers (1D) is 16. Default for Darcy/NS (2D) is 12.
    mode_offsets = [-8, -4, 0, 4, 8]
    
    results = []
    
    for pde, resolutions in pde_configs.items():
        print(f"Evaluating {pde.upper()}")
        
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
            
            # Base defaults
            default_modes = 12 if n_dims == 2 else 16
            width = 32 if n_dims == 2 else 64
            
            for offset in mode_offsets:
                current_modes = default_modes + offset
                if current_modes <= 0:
                    continue # Skip invalid modes
                
                print(f"  -> Training with modes: {current_modes} (offset {offset})...")
                num_modes = (current_modes, current_modes) if n_dims == 2 else (current_modes,)
                model = FNO(modes=num_modes, input_size=in_channels, output_size=out_channels, 
                            n_dims=n_dims, num_layers=4, width=width).to(device)
                
                try:
                    train_time, inf_time, test_err = train_and_eval(model, train_loader, test_loader, args.epochs, device)
                    print(f"     Rel L2 Error: {test_err:.4f}  |  Train Time: {train_time:.2f}s  |  Inf Time: {inf_time:.2f}s")
                    results.append({
                        "PDE": pde,
                        "Resolution": res,
                        "Modes": current_modes,
                        "Mode_Offset": offset,
                        "Rel_L2_Error": test_err,
                        "Train_Time_s": train_time,
                        "Inference_Time_s": inf_time
                    })
                except Exception as e:
                    print(f"[!] Failed to train FNO with modes {current_modes}: {e}")
                    results.append({
                        "PDE": pde,
                        "Resolution": res,
                        "Modes": current_modes,
                        "Mode_Offset": offset,
                        "Rel_L2_Error": np.nan,
                        "Train_Time_s": np.nan,
                        "Inference_Time_s": np.nan
                    })
    
    # Export results
    df = pd.DataFrame(results)
    df.to_csv("experiment_modes_results.csv", index=False)
    print("\nResults saved to experiment_modes_results.csv")

if __name__ == "__main__":
    main()
