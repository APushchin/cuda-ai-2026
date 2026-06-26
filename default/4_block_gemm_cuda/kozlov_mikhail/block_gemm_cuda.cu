#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;
    int phases = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int ph = 0; ph < phases; ++ph) {
        int aCol = ph * BLOCK_SIZE + tx;
        int bRow = ph * BLOCK_SIZE + ty;

        if (row < n && aCol < n) {
            sA[ty][tx] = A[row * n + aCol];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (bRow < n && col < n) {
            sB[ty][tx] = B[bRow * n + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t totalSize = static_cast<size_t>(n) * n;
    size_t bytes = totalSize * sizeof(float);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blockGemmKernel<<<gridDim, blockDim>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> result(totalSize);
    cudaMemcpy(result.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return result;
}
