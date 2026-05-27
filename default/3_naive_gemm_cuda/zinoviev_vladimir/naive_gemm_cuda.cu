#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

#include "naive_gemm_cuda.h"

__global__ void NaiveGemmCUDAKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0.f;
    for(int k = 0; k < n; ++k) {
        res += a[i*n + k] * b[k*n + j];
    }
    c[i*n + j] = res;
}

__global__ void NaiveGemmCUDAKernelCheck(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        float res = 0.f;
        for(int k = 0; k < n; ++k) {
            res += a[i*n + k] * b[k*n + j];
        }
        c[i*n + j] = res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int size = n * n;
    const int memSize = size*sizeof(float);
    std::vector<float> c(size);
    uint block_size;
    uint num_blocks;
    block_size = 32;
    num_blocks = (n + block_size - 1) / block_size;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);
    cudaMemcpy(d_a, a.data(), memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), memSize, cudaMemcpyHostToDevice);
    if (n % block_size == 0) {
        NaiveGemmCUDAKernel<<<{num_blocks, num_blocks}, {block_size, block_size}>>>(d_a, d_b, d_c, n);
    } else {
        NaiveGemmCUDAKernelCheck<<<{num_blocks, num_blocks}, {block_size, block_size}>>>(d_a, d_b, d_c, n);
    }
    cudaMemcpy(c.data(), d_c, memSize, cudaMemcpyDeviceToHost);

    return c;
}
