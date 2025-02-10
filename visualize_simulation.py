import numpy as np
import matplotlib.pyplot as plt

def load_single_file(file_path):
    """
    Reads a single simulation results CSV file and loads the phase field values.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.strip().split(",") if x])
    return np.array(data)

def plot_phase_field(data, step, grid_size):
    """
    Plots the phase field for a specific step and saves it as an image.
    """
    plt.figure(figsize=(8, 8))
    plt.title(f"Phase Field at Step {step}")
    plt.imshow(data, extent=(0, grid_size, 0, grid_size), origin='lower', cmap='coolwarm')
    plt.colorbar(label="Phi Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)

    # Save the figure
    output_filename = f"sequential_phase_field_step_{step}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")

# Define file paths for the steps
files = {
    0: "build/simulation_results_step_0.csv",
    500: "build/simulation_results_step_500.csv",
    1000: "build/simulation_results_step_1000.csv",
    1500: "build/simulation_results_step_1500.csv",
    2000: "build/simulation_results_step_2000.csv"
}

# Grid size (adjust based on simulation)
grid_size = 2048

# Process and plot each file
for step, file_path in files.items():
    print(f"Processing step {step} from file: {file_path}")
    data = load_single_file(file_path)
    plot_phase_field(data, step, grid_size)

