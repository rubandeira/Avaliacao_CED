import numpy as np
import matplotlib.pyplot as plt
import os

def load_simulation_data(file_path):
    """
    Reads a CSV file and returns a NumPy array.
    """
    if not os.path.exists(file_path):
        print(f"⚠ Warning: File {file_path} not found, skipping...")
        return None
    print(f"✅ Loading {file_path}...")
    return np.loadtxt(file_path, delimiter=",")

def plot_phase_field(file_path, step, grid_size):
    """
    Plots and saves the phase field as a high-resolution image.
    """
    phi = load_simulation_data(file_path)
    if phi is None:
        return  # Skip if file is missing

    plt.figure(figsize=(10, 10))
    plt.title(f"Phase Field at Step {step}", fontsize=14)
    plt.imshow(phi, extent=(0, grid_size, 0, grid_size), origin='lower', cmap='coolwarm')
    plt.colorbar(label="Phi Value")
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)

    # Save high-resolution image
    output_filename = f"phase_field_step_{step}.png"
    plt.savefig(output_filename, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"🖼 Saved: {output_filename}")

# Define parameters
grid_size = 2048  # Must match your CUDA simulation grid size
steps_to_plot = [0, 10, 30, 60, 80, 100, 500, 100]  # Customize this based on saved simulation steps

# Process and plot key steps
for step in steps_to_plot:
    file_path = f"/home/alunos/tei/2024/tei26703/AVALIACAO_CED/Avaliacao_CED/build/simulation_results_step_{step}.csv"
    plot_phase_field(file_path, step, grid_size)

print("✅ Visualization completed!")
