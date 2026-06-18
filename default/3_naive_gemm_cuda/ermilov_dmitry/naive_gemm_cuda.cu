#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void NaiveGemmKernel(const float* A,
                                const float* B,
                                float* C,
                                int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;

#pragma unroll
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    NaiveGemmKernel<<<grid, block>>>(dA, dB, dC, n);

    cudaMemcpy(c.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return c;
}