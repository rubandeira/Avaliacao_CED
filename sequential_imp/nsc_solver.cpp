#include "nsc_solver.h"
#include <vector>
#include <cmath>
#include <iostream>

void initializeFields(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& u,
                      std::vector<std::vector<double>>& v,
                      int gridSize, double bubbleRadius) {
    int centerX = gridSize / 2;
    int centerY = gridSize / 2;
    double radiusSquared = bubbleRadius * bubbleRadius;

    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            double dx = i - centerX;
            double dy = j - centerY;
            double distanceSquared = dx * dx + dy * dy;

            // ✅ Assign +1 inside the bubble, -1 outside, and ensure valid values
            phi[i][j] = (distanceSquared <= radiusSquared) ? 1.0 : -1.0;

            // ✅ Prevent uninitialized values
            if (std::isnan(phi[i][j])) {
                std::cerr << "Error: NaN detected in phi initialization at (" << i << ", " << j << ")" << std::endl;
                exit(1);
            }

            u[i][j] = 0.0;
            v[i][j] = 0.0;
        }
    }
}

void updatePhaseField(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params) {
    int N = params.gridSize;
    double epsilonSquared = params.epsilon * params.epsilon;

    std::vector<std::vector<double>> laplacian(N, std::vector<double>(N, 0.0));

    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            laplacian[i][j] = phi[i + 1][j] + phi[i - 1][j] +
                              phi[i][j + 1] + phi[i][j - 1] -
                              4.0 * phi[i][j];

            if (std::isnan(laplacian[i][j])) {
                std::cerr << "Error: NaN detected in laplacian at (" << i << ", " << j << ")" << std::endl;
                exit(1);
            }
        }
    }
    
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double phiVal = phi[i][j];
            double fPrime = 2.0 * phiVal * (1.0 - phiVal * phiVal);
            mu[i][j] = -epsilonSquared * laplacian[i][j] + fPrime;

            if (std::isnan(mu[i][j])) {
                std::cerr << "Error: NaN detected in mu at (" << i << ", " << j << ")" << std::endl;
                exit(1);
            }
        }
    }

    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double muLaplacian = mu[i + 1][j] + mu[i - 1][j] +
                                 mu[i][j + 1] + mu[i][j - 1] - 4.0 * mu[i][j];

            if (std::isnan(muLaplacian)) {
                std::cerr << "Error: NaN detected in muLaplacian at (" << i << ", " << j << ")" << std::endl;
                exit(1);
            }

            phi[i][j] += params.dt * params.mobility * muLaplacian;
        }
    }
}

// ✅ Velocity Update (Navier-Stokes Equations)
void updateVelocityField(std::vector<std::vector<double>>& u,
                         std::vector<std::vector<double>>& v,
                         std::vector<std::vector<double>>& phi,
                         const SimulationParameters& params) {
    int N = params.gridSize;
    double dt = params.dt;
    double rho = params.density;
    double eta = params.viscosity;
    double sigma = params.surfaceTension;

    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double gradPhiX = (phi[i + 1][j] - phi[i - 1][j]) / 2.0;
            double gradPhiY = (phi[i][j + 1] - phi[i][j - 1]) / 2.0;

            double laplacianU = u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4.0 * u[i][j];
            double laplacianV = v[i + 1][j] + v[i - 1][j] + v[i][j + 1] + v[i][j - 1] - 4.0 * v[i][j];

            u[i][j] += dt * (-gradPhiX * sigma / rho + eta * laplacianU / rho);
            v[i][j] += dt * (-gradPhiY * sigma / rho + eta * laplacianV / rho);
        }
    }
}

