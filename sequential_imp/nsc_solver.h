#pragma once

#include <vector>
#include <string>

// Define the simulation parameters
struct SimulationParameters {
    int gridSize;          // Grid resolution
    double dt;             // Time step size
    double simulationTime; // Total simulation time
    double epsilon;        // Interface thickness
    double mobility;       // Mobility parameter
    double density;        // Fluid density
    double viscosity;      // Fluid viscosity
    double surfaceTension; // Surface tension
};

// Function declarations
void initializeFields(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params);

void updateChemicalPotential(std::vector<std::vector<double>>& phi,
                             std::vector<std::vector<double>>& mu,
                             const SimulationParameters& params);

void updatePhaseField(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params);

void saveResults(const std::vector<std::vector<double>>& phi, int step, const std::string& filename);

void runSimulation(const SimulationParameters& params);

