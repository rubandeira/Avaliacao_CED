#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string> 

struct SimulationParameters {
    int gridSize;
    double dt;
    double simulationTime;
    double epsilon;
    double mobility;
    double density;
    double viscosity;
    double surfaceTension;
};

// Declare CUDA functions
void initializeFieldsCUDA(double* d_phi, double* d_u, double* d_v, int gridSize, double bubbleRadius);
void updateChemicalPotentialCUDA(double* d_phi, double* d_mu, SimulationParameters params, int gridSize);
void updatePhaseFieldCUDA(double* d_phi, double* d_mu, SimulationParameters params, int gridSize);
void updateVelocityFieldCUDA(double* d_u, double* d_v, double* d_phi, SimulationParameters params, int gridSize);
void saveResultsFromGPU(const double* d_phi, int gridSize, int step, const std::string& filename);  // ✅ Fix: std::string should now work
