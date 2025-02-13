#include "nsc_solver.h"
#include <chrono>
#include <iostream>

int main() {
    
    SimulationParameters params = {
        .gridSize = 1024,     
        .dt = 0.001,          
        .simulationTime = 2.0, 
        .epsilon = 1.0,       
        .mobility = 10.0,     
        .density = 1.0,       
        .viscosity = 0.01,   
        .surfaceTension = 0.1 
    };

    std::cout << "Starting sequential simulation...\n";

    // Start timing the simulation
    auto start = std::chrono::high_resolution_clock::now();

    // Run the simulation (results will be periodically saved as CSV files)
    runSimulation(params);

    // End timing the simulation
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Simulation completed.\n";
    std::cout << "Execution time: " << elapsed << " seconds.\n";

    return 0;
}

