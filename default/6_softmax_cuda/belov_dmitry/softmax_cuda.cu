#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <stdio.h>

__global__ void softMaxFunc(const float* input, float* output, int colCount)
{
    extern __shared__ float sharedMem[];
    float* sharedMaxSum = sharedMem; 
    float* sharedExp  = &sharedMem[blockDim.x];

    int rowIndex    = blockIdx.x; 
    int threadIndex = threadIdx.x;
    int elStart     = rowIndex * colCount;

    float localMax = -INFINITY;
    for (int iCol = threadIndex; iCol < colCount; iCol += blockDim.x) 
    {
        if (input[elStart + iCol] > localMax)
        {
            localMax = input[elStart + iCol];
        }
    }
    sharedMaxSum[threadIndex] = localMax;
    __syncthreads();

    __shared__ float rowMax;
    if (threadIndex == 0)
    {
        rowMax = sharedMaxSum[0];
        for (int iCol = 1; iCol < blockDim.x; ++iCol) 
        {
            if (sharedMaxSum[iCol] > rowMax)
                rowMax = sharedMaxSum[iCol];
        }
    }
    __syncthreads();

    for (int iCol = threadIndex; iCol < colCount; iCol += blockDim.x) 
    {
        sharedExp[iCol] = expf(input[elStart + iCol] - rowMax);
    }
    __syncthreads();

    float localSum = 0.0f;
    for (int iCol = threadIndex; iCol < colCount; iCol += blockDim.x) 
    {
        localSum += sharedExp[iCol];
    }
    sharedMaxSum[threadIndex] = localSum;
    __syncthreads();

    __shared__ float rowExpSum;
    if (threadIndex == 0)
    {
        rowExpSum = 0;
        for (int iCol = 0; iCol < blockDim.x; ++iCol) 
        {
            rowExpSum += sharedMaxSum[iCol];
        }
    }
    __syncthreads();

    for (int iCol = threadIndex; iCol < colCount; iCol += blockDim.x) 
    {
        output[elStart + iCol] = sharedExp[iCol] / rowExpSum;
    }

}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int rowCount) 
{
    const int mtxNumEl = static_cast<int>(input.size());
    const size_t bitMtxNumEl = mtxNumEl * sizeof(float);
    const int colCount = mtxNumEl / rowCount;

    float *deviceInput = nullptr;
    cudaMalloc(&deviceInput, bitMtxNumEl);
    cudaMemcpy(deviceInput, input.data(), bitMtxNumEl, cudaMemcpyHostToDevice);

    float *deviceOutput = nullptr;
    cudaMalloc(&deviceOutput, bitMtxNumEl);

    const int numThreads = min(256, colCount);
    int numBlocks = rowCount;
    size_t sharedMemSize = (numThreads + colCount) * sizeof(float);
    softMaxFunc<<<numBlocks, numThreads, sharedMemSize>>>(deviceInput, deviceOutput, colCount);
    
    cudaDeviceSynchronize();

    std::vector<float> output(mtxNumEl);
    cudaMemcpy(output.data(), deviceOutput, bitMtxNumEl, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;

}
