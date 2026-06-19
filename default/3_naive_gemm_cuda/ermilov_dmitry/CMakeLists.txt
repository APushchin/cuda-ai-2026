cmake_minimum_required(VERSION 3.10)

project(GeluOMP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


add_executable(naive_gemm_cuda
    main.cpp
    naive_gemm_cuda.cu
)

target_include_directories(naive_gemm_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
