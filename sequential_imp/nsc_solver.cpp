#include "nsc_solver.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

void initializeFields(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params) {
    int N = params.gridSize;
    double radius = 100.0;  

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i - N / 2;
            double y = j - N / 2;
            phi[i][j] = (x * x + y * y <= radius * radius) ? 1.0 : -1.0;
            mu[i][j] = 0.0;
        }
    }
}


void updateChemicalPotential(std::vector<std::vector<double>>& phi,
                             std::vector<std::vector<double>>& mu,
                             const SimulationParameters& params) {
    int N = params.gridSize;
    double epsilonSquared = params.epsilon * params.epsilon;

    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double laplacian = phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1] - 4.0 * phi[i][j];
            mu[i][j] = -phi[i][j] + phi[i][j]*phi[i][j]*phi[i][j] - epsilonSquared * laplacian;
        }
    }
}

void updatePhaseField(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params) {
    int N = params.gridSize;
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double muLaplacian = mu[i+1][j] + mu[i-1][j] + mu[i][j+1] + mu[i][j-1] - 4.0 * mu[i][j];
            phi[i][j] += params.dt * params.mobility * muLaplacian;
        }
    }
}


void saveResults(const std::vector<std::vector<double>>& phi, int step, const std::string& filename) {
    std::ofstream outFile(filename + "_step_" + std::to_string(step) + ".csv");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing!\n";
        return;
    }
    outFile << std::fixed << std::setprecision(6);
    for (const auto& row : phi) {
        for (const auto& value : row) {
            outFile << value << ",";
        }
        outFile << "\n";
    }
    outFile.close();
}


void runSimulation(const SimulationParameters& params) {
    int N = params.gridSize;
    // Calculate total number of steps: simulationTime/dt
    int steps = static_cast<int>(params.simulationTime / params.dt);

    // Allocate and initialize the phase field (phi) and chemical potential (mu)
    std::vector<std::vector<double>> phi(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> mu(N, std::vector<double>(N, 0.0));

    initializeFields(phi, mu, params);

    for (int step = 0; step <= steps; ++step) {
        // Update the chemical potential and phase field
        updateChemicalPotential(phi, mu, params);
        updatePhaseField(phi, mu, params);

       
        if (step % 10 == 0 && step <= 100) {
            saveResults(phi, step, "simulation_results_sequential");
        
        }
    }
}

