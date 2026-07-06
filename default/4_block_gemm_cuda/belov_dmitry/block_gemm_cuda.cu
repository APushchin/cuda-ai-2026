#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void blockGennFunc(const float* mtxA, const float* mtxB, float* mtxC, int mtxSize, int numThreads)
{
    extern __shared__ float sharedBuffer[];
    float* sharedMtxA = sharedBuffer; 
    float* sharedMtxB = &sharedBuffer[numThreads * numThreads];

    int numBlocks = gridDim.x;

    int colIndexBlock = threadIdx.x;
    int rowIndexBlock = threadIdx.y;

    int colIndexFull = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIndexFull = blockIdx.y * blockDim.y + threadIdx.y;

    float mtxCEl = 0.0;

    for (int iBlock = 0; iBlock < numBlocks; ++iBlock)
    {
        if (rowIndexFull < mtxSize && (iBlock*numThreads + colIndexBlock) < mtxSize)
        {
            sharedMtxA[rowIndexBlock*numThreads + colIndexBlock] = 
                mtxA[rowIndexFull*mtxSize + (iBlock*numThreads + colIndexBlock)];
        }
        else
        {
            sharedMtxA[rowIndexBlock*numThreads + colIndexBlock] = 0.0f;
        }


        if ((iBlock*numThreads + rowIndexBlock) < mtxSize && colIndexFull < mtxSize)
        {
            sharedMtxB[rowIndexBlock*numThreads + colIndexBlock] = 
                mtxB[(iBlock*numThreads + rowIndexBlock)*mtxSize + colIndexFull];
        }
        else
        {
            sharedMtxB[rowIndexBlock*numThreads + colIndexBlock] = 0.0f;
        }

        __syncthreads();

        for (int iThread = 0; iThread < numThreads; ++iThread)
            mtxCEl += sharedMtxA[rowIndexBlock*numThreads + iThread] * sharedMtxB[iThread*numThreads + colIndexBlock];
            
        __syncthreads();
    }

    if (rowIndexFull < mtxSize && colIndexFull < mtxSize)
        mtxC[rowIndexFull*mtxSize + colIndexFull] = mtxCEl;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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
    const size_t numMtxElInBlock = numThreads*numThreads;
    const size_t bitMtxNumElNumMtxElInBlock = 2 * numMtxElInBlock * sizeof(float);
    dim3 threadsPerBlock(numThreads, numThreads);
    const size_t numBlocks = (n + numThreads - 1) / numThreads;
    dim3 blockCount(numBlocks, numBlocks);

    blockGennFunc<<<blockCount, threadsPerBlock, bitMtxNumElNumMtxElInBlock>>>(deviceMtxA, deviceMtxB, deviceMtxC, n, numThreads);
    
    cudaDeviceSynchronize();

    std::vector<float> output(mtxNumEl);
    cudaMemcpy(output.data(), deviceMtxC, bitMtxNumEl, cudaMemcpyDeviceToHost);

    cudaFree(deviceMtxA);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxC);

    return output;
}