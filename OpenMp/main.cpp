#include "nsc_solver.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>

int main() {
    SimulationParameters params = {
        .gridSize = 2048,  // Large grid size for better scaling on supercomputer
        .dt = 0.0001,      // Stability-focused time step
        .simulationTime = 1.0,
        .epsilon = 1.0,
        .mobility = 1.0,
        .density = 1.0,
        .viscosity = 0.1,
        .surfaceTension = 1.0
    };

    int gridSize = params.gridSize;
    double bubbleRadius = gridSize / 8.0;

    std::vector<std::vector<double>> phi(gridSize, std::vector<double>(gridSize, 0.0));
    std::vector<std::vector<double>> u(gridSize, std::vector<double>(gridSize, 0.0));
    std::vector<std::vector<double>> v(gridSize, std::vector<double>(gridSize, 0.0));

    initializeFields(phi, u, v, gridSize, bubbleRadius);

    int maxThreads = omp_get_max_threads();
    omp_set_num_threads(maxThreads);
    std::cout << "Running with " << maxThreads << " threads." << std::endl;

    // Open CSV file for output
    std::ofstream outputFile("simulation_results.csv");
    if (!outputFile) {
        std::cerr << "Error: Unable to open simulation_results.csv for writing." << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t <= 100; ++t) {
        std::vector<std::vector<double>> mu(gridSize, std::vector<double>(gridSize, 0.0));
        updatePhaseField(phi, mu, params);
        updateVelocityField(u, v, phi, params);

        // ✅ Save phase field (`phi`) values every 10 steps
        if (t % 10 == 0) {
            outputFile << "Step " << t << "\n";
            for (int i = 0; i < gridSize; ++i) {
                for (int j = 0; j < gridSize; ++j) {
                    outputFile << phi[i][j] << (j == gridSize - 1 ? "\n" : ",");
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Simulation completed in " << elapsed << " seconds." << std::endl;

    outputFile.close();
    return 0;
}

