#include "nsc_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

__global__ void updatePhaseFieldKernel(double* phi, double* mu, SimulationParameters params, int gridSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < gridSize - 1 && j > 0 && j < gridSize - 1) {
        double muLaplacian = mu[(i + 1) * gridSize + j] + mu[(i - 1) * gridSize + j] +
                             mu[i * gridSize + (j + 1)] + mu[i * gridSize + (j - 1)] -
                             4.0 * mu[i * gridSize + j];
        phi[i * gridSize + j] += params.dt * params.mobility * muLaplacian;
    }
}

// ✅ Kernel for initializing the phase field
__global__ void initializeFieldsKernel(double* phi, double* u, double* v, int gridSize, double bubbleRadius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < gridSize && j < gridSize) {
        double x = i - gridSize / 2;
        double y = j - gridSize / 2;
        phi[i * gridSize + j] = (x * x + y * y <= bubbleRadius * bubbleRadius) ? 1.0 : -1.0;
        u[i * gridSize + j] = 0.0;
        v[i * gridSize + j] = 0.0;
    }
}
// ✅ Function to launch kernel
void initializeFieldsCUDA(double* d_phi, double* d_u, double* d_v, int gridSize, double bubbleRadius) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (gridSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeFieldsKernel<<<numBlocks, threadsPerBlock>>>(d_phi, d_u, d_v, gridSize, bubbleRadius);
    cudaDeviceSynchronize();
}
void updatePhaseFieldCUDA(double* d_phi, double* d_mu, SimulationParameters params, int gridSize) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (gridSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    updatePhaseFieldKernel<<<numBlocks, threadsPerBlock>>>(d_phi, d_mu, params, gridSize);
    cudaDeviceSynchronize();
}

// ✅ Kernel for updating chemical potential
__global__ void updateChemicalPotentialKernel(double* phi, double* mu, SimulationParameters params, int gridSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < gridSize - 1 && j > 0 && j < gridSize - 1) {
        double laplacian = phi[(i + 1) * gridSize + j] + phi[(i - 1) * gridSize + j] +
                           phi[i * gridSize + (j + 1)] + phi[i * gridSize + (j - 1)] -
                           4.0 * phi[i * gridSize + j];
        mu[i * gridSize + j] = -phi[i * gridSize + j] + phi[i * gridSize + j] * phi[i * gridSize + j] * phi[i * gridSize + j] - 
                               params.epsilon * params.epsilon * laplacian;
    }
}
// ✅ Define the CUDA kernel for velocity field update
__global__ void updateVelocityFieldKernel(double* u, double* v, double* phi, SimulationParameters params, int gridSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < gridSize && j < gridSize) {
        // Placeholder velocity update (modify based on simulation requirements)
        u[i * gridSize + j] = 0.0;
        v[i * gridSize + j] = 0.0;
    }
}
// ✅ Function to launch the kernel
void updateVelocityFieldCUDA(double* d_u, double* d_v, double* d_phi, SimulationParameters params, int gridSize) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (gridSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    updateVelocityFieldKernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_phi, params, gridSize);
    cudaDeviceSynchronize();
}

// ✅ Wrapper for kernel launch
void updateChemicalPotentialCUDA(double* d_phi, double* d_mu, SimulationParameters params, int gridSize) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (gridSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    updateChemicalPotentialKernel<<<numBlocks, threadsPerBlock>>>(d_phi, d_mu, params, gridSize);
    cudaDeviceSynchronize();
}

// ✅ Function for saving results from GPU
void saveResultsFromGPU(const double* d_phi, int gridSize, int step, const std::string& filename) {
    std::vector<double> h_phi(gridSize * gridSize);
    cudaMemcpy(h_phi.data(), d_phi, gridSize * gridSize * sizeof(double), cudaMemcpyDeviceToHost);

    std::ofstream outFile(filename + "_step_" + std::to_string(step) + ".csv");
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            outFile << h_phi[i * gridSize + j] << (j < gridSize - 1 ? "," : "\n");
        }
    }
}
