#include "nsc_solver.cuh"
#include <iostream>
#include <chrono>

int main() {
    // Define high-resolution simulation parameters
    SimulationParameters params = {
        4096,    // High grid size for detailed graphics
        0.0001,  // Small dt for accuracy
        5.0,     // Long simulation time for detailed evolution
        1.0,     // Interface thickness
        10.0,    // Moderate mobility
        1.0,     // Density remains unchanged
        0.001,   // Low viscosity for fluid-like behavior
        1.0      // Surface tension remains unchanged
    };

int gridSize = params.gridSize;
    double bubbleRadius = gridSize / 8.0;
    int totalSteps = static_cast<int>(params.simulationTime / params.dt);

    // **MULTI-GPU: Check and Assign Devices**
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    numGPUs = std::min(numGPUs, 4);  // Use max 4 GPUs

    if (numGPUs < 4) {
        std::cerr << "Warning: Only " << numGPUs << " GPUs available. Expect lower performance.\n";
    }
    std::cout << "✅ Using " << numGPUs << " NVIDIA A100 GPUs\n";

    // Allocate memory on each GPU
    double *d_phi[4], *d_mu[4], *d_u[4], *d_v[4];
    cudaStream_t streams[4];

    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc((void**)&d_phi[gpu], gridSize * gridSize * sizeof(double));
        cudaMalloc((void**)&d_mu[gpu], gridSize * gridSize * sizeof(double));
        cudaMalloc((void**)&d_u[gpu], gridSize * gridSize * sizeof(double));
        cudaMalloc((void**)&d_v[gpu], gridSize * gridSize * sizeof(double));
        cudaStreamCreate(&streams[gpu]);
    }

    // **Initialize Phase Field on Each GPU**
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        initializeFieldsCUDA(d_phi[gpu], d_u[gpu], d_v[gpu], gridSize, bubbleRadius);
    }

    std::cout << "🚀 Starting Multi-GPU CUDA Simulation...\n";

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // **Run Simulation in Parallel on All GPUs**
    for (int t = 0; t <= totalSteps; ++t) {
        for (int gpu = 0; gpu < numGPUs; gpu++) {
            cudaSetDevice(gpu);
            updateChemicalPotentialCUDA(d_phi[gpu], d_mu[gpu], params, gridSize);
            updatePhaseFieldCUDA(d_phi[gpu], d_mu[gpu], params, gridSize);
            updateVelocityFieldCUDA(d_u[gpu], d_v[gpu], d_phi[gpu], params, gridSize);
        }

        // **Save every 10000 steps from GPU 0**
        if (t % 10000 == 0) {
            saveResultsFromGPU(d_phi[0], gridSize, t, "simulation_results_step");
        }
    }

    // Measure execution time
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "✅ Multi-GPU Simulation Completed in " << elapsed << " seconds.\n";

    // Free GPU memory
    for (int gpu = 0; gpu < numGPUs; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_phi[gpu]);
        cudaFree(d_mu[gpu]);
        cudaFree(d_u[gpu]);
        cudaFree(d_v[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }

    return 0;
}
