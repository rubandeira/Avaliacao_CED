import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_simulation_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        step = None
        grid_data = []
        for line in lines:
            if "Step" in line:
                if step is not None:
                    data[step] = np.array(grid_data)
                step = int(line.split()[1])
                grid_data = []
            else:
                grid_data.append([float(x) for x in line.strip().split(",")])
        if step is not None:
            data[step] = np.array(grid_data)
    return data

def plot_phase_field(data, step, grid_size):
    if step not in data:
        print(f"Step {step} not found in data!")
        return

    phi = data[step]
    plt.figure(figsize=(8, 8))
    plt.title(f"Phase Field at Step {step}")
    plt.imshow(phi, extent=(0, grid_size, 0, grid_size), origin='lower', cmap='coolwarm')
    plt.colorbar(label="Phi Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.show()

# Load simulation results
file_path = "simulation_results.csv"
data = load_simulation_data(file_path)

# Plot results for specific steps
grid_size = 2048  # Update to match your grid size
plot_phase_field(data, step=0, grid_size=grid_size)
plot_phase_field(data, step=50, grid_size=grid_size)
plot_phase_field(data, step=100, grid_size=grid_size)
