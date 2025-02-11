#include "nsc_solver.cuh"
#include <iostream>
#include <chrono>

int main() {
    // Define simulation parameters
    SimulationParameters params = {
        1024,    // gridSize
        0.001,  // dt
        1.0,     // simulationTime
        1.0,     // epsilon
        10.0,    // mobility
        1.0,     // density
        0.01,   // viscosity
        0.1      // surfaceTension
    };

    int gridSize = params.gridSize;
    double bubbleRadius = gridSize / 8.0;

    // Allocate device memory
    double *d_phi, *d_u, *d_v, *d_mu;
    cudaMalloc((void**)&d_phi, gridSize * gridSize * sizeof(double));
    cudaMalloc((void**)&d_u, gridSize * gridSize * sizeof(double));
    cudaMalloc((void**)&d_v, gridSize * gridSize * sizeof(double));
    cudaMalloc((void**)&d_mu, gridSize * gridSize * sizeof(double));

    // Initialize the phase field on GPU
    initializeFieldsCUDA(d_phi, d_u, d_v, gridSize, bubbleRadius);

    int maxThreads;
    cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMultiProcessorCount, 0);
    std::cout << "Running with " << maxThreads << " GPU multiprocessors." << std::endl;

    // Start timing the simulation
    auto start = std::chrono::high_resolution_clock::now();

    int steps = params.simulationTime / params.dt;
    for (int t = 0; t <= steps; ++t) {
        updateChemicalPotentialCUDA(d_phi, d_mu, params, gridSize);
        updatePhaseFieldCUDA(d_phi, d_mu, params, gridSize);
        updateVelocityFieldCUDA(d_u, d_v, d_phi, params, gridSize);


        if (t % 10 == 0 && t <= 100) {
            saveResultsFromGPU(d_phi, gridSize, t, "simulation_results");
        }
    }

    // End timing the simulation
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Simulation completed in " << elapsed << " seconds.\n";

    // Free GPU memory
    cudaFree(d_phi);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_mu);

    return 0;
}

