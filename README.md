#High-Performance Computing Implementation of the Cahn-Hilliard Equation

This repository contains three implementations of a Cahn-Hilliard equation solver, developed as part of a High-Performance Computing (HPC) project. The goal is to analyze the computational efficiency of different parallel computing techniques and compare their performance in solving phase separation problems.

Project Overview

The Cahn-Hilliard equation models phase separation in multi-component systems. Given its computational complexity, three different implementations were developed to explore the benefits of parallelization:

Sequential Implementation → A single-threaded version serving as the baseline for performance comparisons.
OpenMP Implementation → A multi-threaded CPU version leveraging OpenMP to accelerate computations.
CUDA Implementation → A GPU-accelerated version using CUDA to maximize parallel processing efficiency.
Repository Structure

📂 Avaliacao_CED
│── 📂 Sequential     # Baseline implementation (single-threaded)
│── 📂 OpenMP         # Parallelized CPU version using OpenMP
│── 📂 CUDA           # GPU-accelerated version using CUDA
│── 📂 Python         # Scripts for visualization & performance analysis
│── 📜 CMakeLists.txt # Build configuration for CMake
│── 📜 README.md      # This file
Compilation & Execution

This project uses CMake for streamlined compilation. Follow these steps to build and execute the code:

# Create a build directory and compile the project
mkdir build && cd build
cmake .. && make -j$(nproc)

# Run the different implementations
./Sequential/sequential_imp
./OpenMP/openmp_imp
./CUDA/cuda_imp
Performance Analysis & Results

Execution time for each implementation was measured, and speedup factors were computed to assess efficiency:

Implementation	Execution Time (s)	Speedup
Sequential	135.2	1.0×
OpenMP	58.7	2.3×
CUDA	2.6	52.0×
The results demonstrate that CUDA massively outperforms CPU-based approaches, highlighting the efficiency of GPU parallelization for large-scale simulations.

Visualization & Speedup Analysis

Python scripts are included to generate visual representations of the phase field evolution and performance scaling. Run them with:

python3 Python/visualize_simulation.py
python3 Python/speedup_analysis.py
These scripts generate graphs showing execution time comparisons, speedup factors, and phase evolution over time.

Future Work & Improvements

Implementing multi-GPU scaling to further optimize CUDA performance.
Testing adaptive mesh refinement (AMR) for enhanced accuracy.
Exploring hybrid CPU-GPU execution to balance workload distribution.
Contributors

Rúben Bandeira
