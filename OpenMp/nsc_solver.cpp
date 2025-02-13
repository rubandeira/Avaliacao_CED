#include "nsc_solver.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>

// Initialize Fields (Parallelized)
void initializeFields(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& u,
                      std::vector<std::vector<double>>& v,
                      int gridSize, double bubbleRadius) {
    int centerX = gridSize / 2;
    int centerY = gridSize / 2;
    double radiusSquared = bubbleRadius * bubbleRadius;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            double dx = i - centerX;
            double dy = j - centerY;
            double distanceSquared = dx * dx + dy * dy;

            phi[i][j] = (distanceSquared <= radiusSquared) ? 1.0 : -1.0;
            u[i][j] = 0.0;
            v[i][j] = 0.0;
        }
    }
}

// Update Phase Field (Cahn-Hilliard with Tiling)
void updatePhaseField(std::vector<std::vector<double>>& phi,
                      std::vector<std::vector<double>>& mu,
                      const SimulationParameters& params) {
    int N = params.gridSize;
    double epsilonSquared = params.epsilon * params.epsilon;

    const int TILE_SIZE = 64; 

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 1; ii < N - 1; ii += TILE_SIZE) {
        for (int jj = 1; jj < N - 1; jj += TILE_SIZE) {
            for (int i = ii; i < std::min(ii + TILE_SIZE, N - 1); ++i) {
                for (int j = jj; j < std::min(jj + TILE_SIZE, N - 1); ++j) {
                    double laplacian = phi[i + 1][j] + phi[i - 1][j] +
                                       phi[i][j + 1] + phi[i][j - 1] -
                                       4.0 * phi[i][j];
                    double fPrime = 2.0 * phi[i][j] * (1.0 - phi[i][j] * phi[i][j]);
                    mu[i][j] = -epsilonSquared * laplacian + fPrime;
                }
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 1; ii < N - 1; ii += TILE_SIZE) {
        for (int jj = 1; jj < N - 1; jj += TILE_SIZE) {
            for (int i = ii; i < std::min(ii + TILE_SIZE, N - 1); ++i) {
                for (int j = jj; j < std::min(jj + TILE_SIZE, N - 1); ++j) {
                    double muLaplacian = mu[i + 1][j] + mu[i - 1][j] +
                                         mu[i][j + 1] + mu[i][j - 1] -
                                         4.0 * mu[i][j];
                    phi[i][j] += params.dt * params.mobility * muLaplacian;
                }
            }
        }
    }
}


void updateVelocityField(std::vector<std::vector<double>>& u,
                         std::vector<std::vector<double>>& v,
                         std::vector<std::vector<double>>& phi,
                         const SimulationParameters& params) {
    int N = params.gridSize;
    double dt = params.dt;
    double rho = params.density;
    double eta = params.viscosity;
    double sigma = params.surfaceTension;

    #pragma omp parallel for collapse(2)
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

