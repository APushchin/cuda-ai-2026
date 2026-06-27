#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <iostream>

using DataType = __half;
inline constexpr int kWmmaSize = 16;

// Anonymous namespace
namespace
{
    __global__ void convertFloatToHalf(const float* __restrict__ input, DataType* __restrict__ output, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
        output[idx] = __float2half(input[idx]);
        }
    }
    
    __global__ void blockGemmImpl(const DataType* __restrict__ a, const DataType* __restrict__ b, float* __restrict__ c, int n) 
    {
        int warpI = (blockIdx.x * 4) + (threadIdx.x >> 5);
        int warpJ = blockIdx.y;
        
        int cRow = warpI << 4;
        int cCol = warpJ << 4;
        
        if (cRow >= n || cCol >= n)
            return;
        
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaSize, kWmmaSize, kWmmaSize, DataType, nvcuda::wmma::row_major> aFrag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaSize, kWmmaSize, kWmmaSize, DataType, nvcuda::wmma::row_major> bFrag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaSize, kWmmaSize, kWmmaSize, float> accFrag;
        
        nvcuda::wmma::fill_fragment(accFrag, 0.0f);
        
        int aTileOffset = cRow * n;
        int bTileOffset = cCol;
        
        bool fullBlock = (cRow + kWmmaSize <= n) && (cCol + kWmmaSize <= n);
        
        for (int k = 0; k < n; k += kWmmaSize)
        {
            if (fullBlock || (k + kWmmaSize <= n))
            {
                nvcuda::wmma::load_matrix_sync(aFrag, a + aTileOffset + k, n);
                nvcuda::wmma::load_matrix_sync(bFrag, b + bTileOffset + k * n, n);
                nvcuda::wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
            }
        }
        
        if (cRow + kWmmaSize <= n && cCol + kWmmaSize <= n)
            nvcuda::wmma::store_matrix_sync(c + aTileOffset + cCol, accFrag, n, nvcuda::wmma::mem_row_major);
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    // Place your implementation here
    const int lenVec = n * n;
    const int sizeFloat = lenVec * sizeof(float);
    const int sizeHalf = lenVec * sizeof(DataType);
    
    float* aFloatDev = nullptr;
    float* bFloatDev = nullptr;
    DataType* aHalfDev = nullptr;
    DataType* bHalfDev = nullptr;
    float* cDev = nullptr;
    
    cudaMalloc(&aFloatDev, sizeFloat);
    cudaMalloc(&bFloatDev, sizeFloat);
    cudaMalloc(&aHalfDev, sizeHalf);
    cudaMalloc(&bHalfDev, sizeHalf);
    cudaMalloc(&cDev, sizeFloat);
    
    cudaMemcpy(aFloatDev, a.data(), sizeFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(bFloatDev, b.data(), sizeFloat, cudaMemcpyHostToDevice);
    
    constexpr int blockSize = 128;
    int gridSize = (lenVec + blockSize - 1) / blockSize;
    convertFloatToHalf<<<gridSize, blockSize>>>(aFloatDev, aHalfDev, lenVec);
    convertFloatToHalf<<<gridSize, blockSize>>>(bFloatDev, bHalfDev, lenVec);
    
    dim3 gridDim((n + kWmmaSize - 1) / kWmmaSize, (n + kWmmaSize - 1) / kWmmaSize, 1);
    dim3 blockDim(blockSize, 1, 1);
    
    blockGemmImpl<<<gridDim, blockDim>>>(aHalfDev, bHalfDev, cDev, n);
    
    cudaDeviceSynchronize();
    
    std::vector<float> result(lenVec);
    cudaMemcpy(result.data(), cDev, sizeFloat, cudaMemcpyDeviceToHost);
    
    cudaFree(aFloatDev);
    cudaFree(bFloatDev);
    cudaFree(aHalfDev);
    cudaFree(bHalfDev);
    cudaFree(cDev);
    
    return result;
}
