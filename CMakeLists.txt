cmake_minimum_required(VERSION 3.22)
project(AVALIACAO_CED LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)




# Sequential Implementation
add_executable(sequential_imp
    sequential_imp/main.cpp
    sequential_imp/nsc_solver.cpp
)
target_include_directories(sequential_imp PRIVATE ${CMAKE_SOURCE_DIR}/sequential_imp)

# OpenMP Implementation
find_package(OpenMP REQUIRED)
add_executable(OpenMp
    OpenMp/main.cpp
    OpenMp/nsc_solver.cpp
)
target_include_directories(OpenMp PRIVATE ${CMAKE_SOURCE_DIR}/OpenMp)
target_link_libraries(OpenMp PRIVATE OpenMP::OpenMP_CXX)

# CUDA Implementation
add_executable(GPU_cuda
    GPU_cuda/main.cu
    GPU_cuda/nsc_solver.cu
)
set_target_properties(GPU_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(GPU_cuda PRIVATE ${CMAKE_SOURCE_DIR}/GPU_cuda)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 --use_fast_math")
