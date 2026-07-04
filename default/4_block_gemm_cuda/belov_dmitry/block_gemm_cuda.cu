#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void blockGennFunc(const float* mtxA, const float* mtxB, float* mtxC, int mtxSize, int numThreads)
{
    extern __shared__ float localMtxA[];
    extern __shared__ float localMtxB[];

    int numBlocks = gridDim.x;

    int colIndexBlock = threadIdx.x;
    int rowIndexBlock = threadIdx.y;

    int colIndexFull = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIndexFull = blockIdx.y * blockDim.y + threadIdx.y;

    float mtxCEl = 0.0;

    for (int iBlock = 0; iBlock < numBlocks; ++iBlock)
    {
        localMtxA[colIndexBlock*numThreads + rowIndexBlock] = 
            mtxA[rowIndexFull*mtxSize + (iBlock*numThreads + colIndexBlock)];

        localMtxB[colIndexBlock*numThreads + rowIndexBlock] = 
            mtxB[(iBlock*numThreads + rowIndexBlock)*mtxSize + colIndexFull];

        __syncthreads();

        for (int iThread = 0; iThread < numThreads; ++iThread)
            mtxCEl += localMtxA[rowIndexBlock*numThreads + iThread] * localMtxB[iThread*numThreads + colIndexBlock];

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
    const size_t bitMtxNumElNumMtxElInBlock = numMtxElInBlock * sizeof(float);
    dim3 threadsPerBlock(numThreads, numThreads);
    const size_t numBlocks = (mtxNumEl + numThreads - 1) / numThreads;
    dim3 blockCount(numBlocks, numBlocks);
    blockGennFunc<<<blockCount, threadsPerBlock, bitMtxNumElNumMtxElInBlock>>>(deviceMtxA, deviceMtxB, deviceMtxC, n, numThreads);

    std::vector<float> output(mtxNumEl);
    cudaMemcpy(output.data(), deviceMtxC, bitMtxNumEl, cudaMemcpyDeviceToHost);

    cudaFree(deviceMtxC);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxA);

    return output;
}