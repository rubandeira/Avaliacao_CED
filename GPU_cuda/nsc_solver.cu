#include "nsc_solver.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void updateField(double* field, int gridSize, double dt, double epsilon, double mobility) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < gridSize - 1 && j > 0 && j < gridSize - 1) {
        int idx = i * gridSize + j;
        field[idx] += dt * (epsilon - mobility * field[idx]);
    }
}

void runGPU(int gridSize, double dt, double simulationTime, double epsilon, double mobility) {
    int timeSteps = static_cast<int>(simulationTime / dt);
    size_t size = gridSize * gridSize * sizeof(double);

    // Allocate memory on the host and GPU
    double* h_field = new double[gridSize * gridSize];
    double* d_field;
    cudaMalloc(&d_field, size);

    // Initialize the field on the host
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            h_field[i * gridSize + j] = (i + j) % 2 == 0 ? 1.0 : -1.0;
        }
    }

    // Copy the field to the GPU
    cudaMemcpy(d_field, h_field, size, cudaMemcpyHostToDevice);

    // Define CUDA kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize2D((gridSize + blockSize.x - 1) / blockSize.x,
                    (gridSize + blockSize.y - 1) / blockSize.y);

    // Run the simulation
    for (int t = 0; t < timeSteps; ++t) {
        updateField<<<gridSize2D, blockSize>>>(d_field, gridSize, dt, epsilon, mobility);
        cudaDeviceSynchronize();
    }

    // Copy the results back to the host
    cudaMemcpy(h_field, d_field, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_field);

    // Free host memory
    delete[] h_field;

    std::cout << "GPU simulation completed successfully using CUDA." << std::endl;
}
