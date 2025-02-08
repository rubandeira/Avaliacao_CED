#include "nsc_solver.h"
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    SimulationParameters params = {
        .gridSize = 128,
        .dt = 0.0001,
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

    int steps = static_cast<int>(params.simulationTime / params.dt);
    for (int t = 0; t < steps; ++t) {
        std::vector<std::vector<double>> mu(gridSize, std::vector<double>(gridSize, 0.0));
        updatePhaseField(phi, mu, params);
        updateVelocityField(u, v, phi, params);
    }
	std::ofstream outFile("simulation_results.txt");
	if (outFile.is_open()) {
	    outFile << "i j phi[i][j]\n";  // Add headers for clarity
	    for (int i = 0; i < gridSize; ++i) {
	        for (int j = 0; j < gridSize; ++j) {
	            outFile << i << " " << j << " " << phi[i][j] << "\n";
	        }
	    }
	    outFile.close();
	    std::cout << "Simulation results saved to simulation_results.txt" << std::endl;
	} else {
	    std::cerr << "Error: Unable to open file for writing." << std::endl;
	}
    std::cout << "Simulation complete." << std::endl;
    return 0;
}

