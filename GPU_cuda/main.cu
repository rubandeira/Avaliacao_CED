#include "nsc_solver.h"
#include <iostream>
#include <chrono>

int main() {
    // Simulation parameters
    int gridSize = 128;
    double dt = 0.01;
    double simulationTime = 1.0;
    double epsilon = 1.0;
    double mobility = 1.0;

    std::cout << "CUDA Navier-Stokes Cahn-Hilliard Simulation" << std::endl;

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    runGPU(gridSize, dt, simulationTime, epsilon, mobility);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Execution time: " << elapsed << " seconds." << std::endl;

    return 0;
}
