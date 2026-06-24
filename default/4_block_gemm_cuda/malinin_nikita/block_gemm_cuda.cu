#include "block_gemm_cuda.h"
#include <cuda/cmath>

#define BLOCK_SIZE 16
#define BLOCK_SIZE_DOUBLE BLOCK_SIZE * BLOCK_SIZE

__global__ void BlockGemmCUDAImpl(const float *a, const float *b, float *c, int n) {
    __shared__ int blockA[BLOCK_SIZE_DOUBLE];
    __shared__ int blockB[BLOCK_SIZE_DOUBLE];

    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int blockSize = blockDim.x;

    int x = blockIdx.x * blockSize + t_x;
    int y = blockIdx.y * blockSize + t_y;
    int blockAB = t_y * blockSize + t_x;

    float sum = 0.0f;

    for (int block = 0; block < gridDim.x; ++block) {
        blockA[blockAB] = a[y * n + block * blockSize + t_x];
        blockB[blockAB] = b[(block * blockSize + t_y) * n + x];
        __syncthreads();
    }


    for (int k = 0; k < blockSize; ++k) {
        sum += blockA[t_y * blockSize + k] * blockB[k * blockSize + t_x];
    }

    __syncthreads();
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = a.size();
    int bytesSize = n * n * sizeof(float);

    float* gpuBufferA = nullptr;
    float* gpuBufferB = nullptr;
    float* gpuBufferC = nullptr;

    cudaMalloc(&gpuBufferA, bytesSize);
    cudaMalloc(&gpuBufferB, bytesSize);
    cudaMalloc(&gpuBufferC, bytesSize);

    cudaMemcpy(gpuBufferA, a.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuBufferB, b.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemset(gpuBufferC, 0, bytesSize);

    int blocks = n / BLOCK_SIZE;
    dim3 threadsDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksDim(blocks, blocks);

    BlockGemmCUDAImpl<<<blocksDim, threadsDim>>>(gpuBufferA, gpuBufferB, gpuBufferC, n);

    std::vector<float> c(size);
    float* cData = c.data();

    cudaDeviceSynchronize();
    cudaMemcpy(cData, gpuBufferC, bytesSize, cudaMemcpyDeviceToHost);
    cudaFree(gpuBufferA);
    cudaFree(gpuBufferB);
    cudaFree(gpuBufferC);

    return c;
}
