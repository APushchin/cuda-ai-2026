cmake_minimum_required(VERSION 3.10)

project(1_gelu_omp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(1_gelu_omp main.cpp gelu_omp.cpp)
target_link_libraries(1_gelu_omp PRIVATE OpenMP::OpenMP_CXX)
