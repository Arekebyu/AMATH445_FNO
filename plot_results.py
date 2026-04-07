import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiment_modes_results.csv')

plt.figure(figsize=(15, 5))
for i, pde in enumerate(df['PDE'].unique(), 1):
    plt.subplot(1, 3, i)
    pde_df = df[df['PDE'] == pde]
    for res in pde_df['Resolution'].unique():
        subset = pde_df[pde_df['Resolution'] == res].sort_values('Modes')
        plt.plot(subset['Modes'], subset['Rel_L2_Error'], marker='o', label=f'Res {res}')
    plt.title(f"{pde.capitalize()} - Error vs Modes")
    plt.yscale('log')
    plt.xlabel('Modes')
    plt.ylabel('Rel L2 Error')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('error_vs_modes.png', dpi=200)
plt.close()

plt.figure(figsize=(15, 5))
for i, pde in enumerate(df['PDE'].unique(), 1):
    plt.subplot(1, 3, i)
    pde_df = df[df['PDE'] == pde]
    for res in pde_df['Resolution'].unique():
        subset = pde_df[pde_df['Resolution'] == res].sort_values('Modes')
        plt.plot(subset['Modes'], subset['Train_Time_s'], marker='o', label=f'Res {res}')
    plt.title(f"{pde.capitalize()} - Train Time vs Modes")
    plt.xlabel('Modes')
    plt.ylabel('Train Time (s)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('time_vs_modes.png', dpi=200)
plt.close()
