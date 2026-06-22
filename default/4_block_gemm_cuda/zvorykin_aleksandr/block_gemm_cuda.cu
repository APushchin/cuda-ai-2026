#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdlib>

// CUDA constants
constexpr int s_BlockM  = 64;
constexpr int s_BlockN  = 64;
constexpr int s_BlockK  = 32;
constexpr int s_TileM   = 4;
constexpr int s_TileN   = 8;
constexpr int s_Pad     = 1;

// Anonymous namespace
namespace
{
    __global__ void optBlockGemmImpl(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     int n)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int rowBase = blockIdx.y * s_BlockM;
        const int colBase = blockIdx.x * s_BlockN;

        if (rowBase >= n || colBase >= n) return;

        __shared__ float as0[s_BlockK][s_BlockM + s_Pad];
        __shared__ float as1[s_BlockK][s_BlockM + s_Pad];
        __shared__ float bs0[s_BlockK][s_BlockN + s_Pad];
        __shared__ float bs1[s_BlockK][s_BlockN + s_Pad];

        float (*asCur)[s_BlockM + s_Pad] = as0;
        float (*asNxt)[s_BlockM + s_Pad] = as1;
        float (*bsCur)[s_BlockN + s_Pad] = bs0;
        float (*bsNxt)[s_BlockN + s_Pad] = bs1;

        float acc[s_TileM * s_TileN];
        #pragma unroll
        for (int i = 0; i < s_TileM * s_TileN; ++i)
            acc[i] = 0.0f;

        const int localRow = ty * s_TileM;
        const int localCol = tx * s_TileN;

        #pragma unroll
        for (int k = 0; k < s_BlockK; ++k)
        {
            const int aRow = rowBase + tx * 8;
            if (aRow + 7 < n)
            {
                float4 aVals = *reinterpret_cast<const float4*>(a + aRow * n + k);
                asCur[k][aRow + 0] = aVals.x; asCur[k][aRow + 1] = aVals.y;
                asCur[k][aRow + 2] = aVals.z; asCur[k][aRow + 3] = aVals.w;
            }
            #pragma unroll
            for (int i = 4; i < 8; ++i)
            {
                if (aRow + i < n) asCur[k][aRow + i] = a[(aRow + i) * n + k];
            }

            const int bCol = colBase + ty * 16;
            if (bCol + 15 < n)
            {
                float4 bVals = *reinterpret_cast<const float4*>(b + k * n + bCol);
                bsCur[k][bCol + 0] = bVals.x; bsCur[k][bCol + 1] = bVals.y;
                bsCur[k][bCol + 2] = bVals.z; bsCur[k][bCol + 3] = bVals.w;
            }
            #pragma unroll
            for (int j = 4; j < 16; ++j)
            {
                if (bCol + j < n) bsCur[k][bCol + j] = b[k * n + bCol + j];
            }
        }
        __syncthreads();

        for (int bk = s_BlockK; bk < n; bk += s_BlockK)
        {
            #pragma unroll
            for (int k = 0; k < s_BlockK; ++k)
            {
                const int aRow = rowBase + tx * 8 + k;
                if (aRow < n)
                {
                    float4 aVals = *reinterpret_cast<const float4*>(a + aRow * n + bk);
                    asNxt[k][aRow + 0] = aVals.x; asNxt[k][aRow + 1] = aVals.y;
                    asNxt[k][aRow + 2] = aVals.z; asNxt[k][aRow + 3] = aVals.w;
                }
                for (int i = 4; i < 8; ++i)
                {
                    if (aRow + i < n) asNxt[k][aRow + i] = a[(aRow + i) * n + bk];
                }

                const int bCol = colBase + ty * 16 + k;
                if (bCol < n)
                {
                    float4 bVals = *reinterpret_cast<const float4*>(b + (bk) * n + bCol);
                    bsNxt[k][bCol + 0] = bVals.x; bsNxt[k][bCol + 1] = bVals.y;
                    bsNxt[k][bCol + 2] = bVals.z; bsNxt[k][bCol + 3] = bVals.w;
                }
                for (int j = 4; j < 16; ++j)
                {
                    if (bCol + j < n) bsNxt[k][bCol + j] = b[(bk) * n + bCol + j];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < s_BlockK; ++k)
            {
                #pragma unroll
                for (int i = 0; i < s_TileM; ++i)
                {
                    float aReg = asCur[k][rowBase + localRow + i];
                    #pragma unroll
                    for (int j = 0; j < s_TileN; ++j)
                    {
                        float bReg = bsCur[k][colBase + localCol + j];
                        acc[i * s_TileN + j] += aReg * bReg;
                    }
                }
            }
            __syncthreads();

            float (*tmpA)[s_BlockM + s_Pad] = asCur;
            asCur = asNxt; asNxt = tmpA;
            float (*tmpB)[s_BlockN + s_Pad] = bsCur;
            bsCur = bsNxt; bsNxt = tmpB;
        }

        int remK = n % s_BlockK;
        if (remK > 0)
        {
            #pragma unroll
            for (int k = 0; k < remK; ++k)
            {
                const int aRow = rowBase + tx * 8 + k;
                if (aRow < n) 
                  asCur[k][aRow] = a[aRow * n + remK + k - remK];
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < remK; ++k)
            {
                #pragma unroll
                for (int i = 0; i < s_TileM; ++i)
                {
                    float aReg = asCur[k][rowBase + localRow + i];
                    #pragma unroll
                    for (int j = 0; j < s_TileN; ++j)
                    {
                        float bReg = bsCur[k][colBase + localCol + j];
                        acc[i * s_TileN + j] += aReg * bReg;
                    }
                }
            }
        }
        
        #pragma unroll
        for (int i = 0; i < s_TileM; ++i)
        {
            #pragma unroll
            for (int j = 0; j < s_TileN; ++j)
            {
                int globalRow = rowBase + localRow + i;
                int globalCol = colBase + localCol + j;
                if (globalRow < n && globalCol < n)
                {
                    c[globalRow * n + globalCol] = acc[i * s_TileN + j];
                }
            }
        }
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    // Place your implementation here
    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    
    float *pinnedA = nullptr;
    float *pinnedB = nullptr;
    float *pinnedC = nullptr;
    
    cudaHostAlloc(&pinnedA, bytes, cudaHostAllocMapped);
    cudaHostAlloc(&pinnedB, bytes, cudaHostAllocMapped);
    cudaHostAlloc(&pinnedC, bytes, cudaHostAllocMapped);

    std::memcpy(pinnedA, a.data(), bytes);
    std::memcpy(pinnedB, b.data(), bytes);
    std::memset(pinnedC, 0, bytes);

    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(deviceA, pinnedA, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceB, pinnedB, bytes, cudaMemcpyHostToDevice, stream);

    dim3 blockDim(s_TileN, s_TileM);
    dim3 gridDim((n + s_BlockN - 1) / s_BlockN, (n + s_BlockM - 1) / s_BlockM);
    optBlockGemmImpl<<<gridDim, blockDim, 0, stream>>>(deviceA, deviceB, deviceC, n);

    cudaMemcpyAsync(pinnedC, deviceC, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<float> c(n * n);
    std::memcpy(c.data(), pinnedC, bytes);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(pinnedA);
    cudaFreeHost(pinnedB);
    cudaFreeHost(pinnedC);

    return c;
}
