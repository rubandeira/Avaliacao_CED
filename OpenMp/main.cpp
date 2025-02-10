#include "nsc_solver.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>

int main() {
    // Define simulation parameters (matching the sequential version)
    SimulationParameters params = {
        .gridSize = 1024,   // Same grid size as sequential
        .dt = 0.001,       // Same time step
        .simulationTime = 2.0, // Same total simulation time
        .epsilon = 1.0,     // Interface thickness
        .mobility = 10.0,   // Mobility for phase evolution
        .density = 1.0,     // Density remains unchanged
        .viscosity = 0.01, // Low viscosity for less damping
        .surfaceTension = 1.0 // Surface tension unchanged
    };

    int gridSize = params.gridSize;
    int steps = static_cast<int>(params.simulationTime / params.dt); // Ensure same step count as sequential

    // Initialize fields
    std::vector<std::vector<double>> phi(gridSize, std::vector<double>(gridSize, 0.0));
    std::vector<std::vector<double>> mu(gridSize, std::vector<double>(gridSize, 0.0));
    
    // Parallelize initialization
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            double x = i - gridSize / 2;
            double y = j - gridSize / 2;
            phi[i][j] = (x * x + y * y <= (gridSize / 8.0) * (gridSize / 8.0)) ? 1.0 : -1.0;
            mu[i][j] = 0.0;
        }
    }

    int maxThreads = omp_get_max_threads();
    omp_set_num_threads(maxThreads);
    std::cout << "Running OpenMP with " << maxThreads << " threads.\n";

    auto start = std::chrono::high_resolution_clock::now();

    // **Ensure same number of steps as Sequential Version**
    for (int t = 0; t <= steps; ++t) {

        // **Parallel Phase Field Update**
        #pragma omp parallel
        {
            #pragma omp for collapse(2)
            for (int i = 1; i < gridSize - 1; ++i) {
                for (int j = 1; j < gridSize - 1; ++j) {
                    double laplacian = phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1] - 4.0 * phi[i][j];
                    mu[i][j] = -phi[i][j] + phi[i][j]*phi[i][j]*phi[i][j] - params.epsilon * params.epsilon * laplacian;
                }
            }

            // **Update Phase Field Using the Chemical Potential**
            #pragma omp for collapse(2)
            for (int i = 1; i < gridSize - 1; ++i) {
                for (int j = 1; j < gridSize - 1; ++j) {
                    double muLaplacian = mu[i+1][j] + mu[i-1][j] + mu[i][j+1] + mu[i][j-1] - 4.0 * mu[i][j];
                    phi[i][j] += params.dt * params.mobility * muLaplacian;
                }
            }
        }

        // ✅ Save phase field (`phi`) values every **500** steps (same as Sequential)
        if (t % 500 == 0) {
            std::string filename = "simulation_results_step_" + std::to_string(t) + ".csv";
            std::ofstream outputFile(filename);
            if (!outputFile) {
                std::cerr << "Error: Unable to open " << filename << " for writing!\n";
                return 1;
            }

            for (int i = 0; i < gridSize; ++i) {
                for (int j = 0; j < gridSize; ++j) {
                    outputFile << phi[i][j] << (j == gridSize - 1 ? "\n" : ",");
                }
            }
            outputFile.close();
            std::cout << "Saved results to " << filename << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Simulation completed in " << elapsed << " seconds.\n";

    return 0;
}

