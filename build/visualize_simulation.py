import numpy as np
import matplotlib.pyplot as plt

def load_simulation_data(file_path):
    """
    Reads the simulation results CSV file and stores phase field values for different steps.
    """
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
    """
    Plots the phase field for a specific step and saves it as an image.
    """
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

    # ✅ Save the figure instead of displaying it
    output_filename = f"phase_field_step_{step}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")

# Load simulation results
file_path = "simulation_results.csv"
data = load_simulation_data(file_path)

# Set grid size (must match what was used in simulation)
grid_size = 2048  # Change based on the actual simulation size

# Generate plots for selected steps
plot_phase_field(data, step=0, grid_size=grid_size)
plot_phase_field(data, step=50, grid_size=grid_size)
plot_phase_field(data, step=100, grid_size=grid_size)

