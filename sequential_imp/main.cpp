#include "nsc_solver.h"
#include <chrono>
#include <iostream>

int main() {
    // Define simulation parameters with adjustments for faster runtime while maintaining physical relevance
    SimulationParameters params = {
        .gridSize = 1024,     // Reduced grid resolution to balance runtime and accuracy
        .dt = 0.001,          // Larger time step to speed up computation
        .simulationTime = 2.0, // Reduced simulation time
        .epsilon = 1.0,       // Interface thickness remains the same
        .mobility = 10.0,     // Adjusted mobility for phase evolution
        .density = 1.0,       // Fluid density unchanged
        .viscosity = 0.01,    // Slightly higher viscosity for numerical stability
        .surfaceTension = 1.0 // Surface tension unchanged
    };

    std::cout << "Starting sequential simulation...\n";

    // Start timing the simulation
    auto start = std::chrono::high_resolution_clock::now();

    // Run the simulation (results will be periodically saved as CSV files)
    runSimulation(params);

    // End timing the simulation
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Simulation completed.\n";
    std::cout << "Execution time: " << elapsed << " seconds.\n";

    return 0;
}

