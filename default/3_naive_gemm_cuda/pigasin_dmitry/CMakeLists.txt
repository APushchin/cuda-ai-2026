cmake_minimum_required(VERSION 3.10)

project(3_naive_gemm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(3_naive_gemm_cuda main.cpp naive_gemm_cuda.cu)
