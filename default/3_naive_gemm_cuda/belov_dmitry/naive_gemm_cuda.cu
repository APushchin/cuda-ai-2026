#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void naiveGennFunc(const float* mtxA, const float* mtxB, float* mtxC, int mtxSize)
{
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (colIndex < mtxSize && rowIndex < mtxSize)
    {
        int tmpIndex = rowIndex * mtxSize;
        float resElC = 0.0f;
        for (int i = 0; i < mtxSize; ++i)
            resElC += mtxA[tmpIndex + i] * mtxB[i * mtxSize + colIndex];

        mtxC[tmpIndex + colIndex] = resElC;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) 
{
    const int mtxNumEl = static_cast<int>(a.size());
    const size_t bitMtxNumEl = mtxNumEl * sizeof(float);

    float *deviceMtxA = nullptr;
    cudaMalloc(&deviceMtxA, bitMtxNumEl);
    cudaMemcpy(deviceMtxA, a.data(), bitMtxNumEl, cudaMemcpyHostToDevice);

    float *deviceMtxB = nullptr;
    cudaMalloc(&deviceMtxB, bitMtxNumEl);
    cudaMemcpy(deviceMtxB, b.data(), bitMtxNumEl, cudaMemcpyHostToDevice);

    float *deviceMtxC = nullptr;
    cudaMalloc(&deviceMtxC, bitMtxNumEl);

    const size_t numThreads = 16;
    dim3 threadsPerBlock(numThreads, numThreads);
    const size_t numBlocks = (n + numThreads - 1) / numThreads;
    dim3 blockCount(numBlocks, numBlocks);

    naiveGennFunc<<<blockCount, threadsPerBlock>>>(deviceMtxA, deviceMtxB, deviceMtxC, n);
    
    cudaDeviceSynchronize();

    std::vector<float> output(mtxNumEl);
    cudaMemcpy(output.data(), deviceMtxC, bitMtxNumEl, cudaMemcpyDeviceToHost);

    cudaFree(deviceMtxA);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxC);

    return output;
}