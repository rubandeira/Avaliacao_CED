#ifndef NSC_SOLVER_H
#define NSC_SOLVER_H

#include <vector>


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


void initializeFields(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& u,
                      std::vector<std::vector<double>>& v,
                      int gridSize, double bubbleRadius);

void updatePhaseField(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params);

void updateVelocityField(std::vector<std::vector<double>>& u,
                         std::vector<std::vector<double>>& v,
                         std::vector<std::vector<double>>& phi,
                         const SimulationParameters& params);

#endif // NSC_SOLVER_H

