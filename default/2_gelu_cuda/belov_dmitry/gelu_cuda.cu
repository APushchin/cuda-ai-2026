#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <math_constants.h>

namespace
{
    constexpr float multVal1 = 1.5957691216; 
    constexpr float multVal2 = 0.044715;

    __global__ void geluFunc(float* dataInOut, int length)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < length)
        {
            const float dataIn      = dataInOut[index];
            const float tanhArg     = multVal1 * (dataIn + multVal2*dataIn*dataIn*dataIn);
            const float expVal      = cuda::std::expf(tanhArg);
            const float tanhVal     = (expVal - 1)/(expVal + 1);
            const float dataOut     = dataIn/2 * (1 + tanhVal);
            dataInOut[index]        = dataOut;
        }
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    const size_t size = static_cast<int>(input.size());
    const size_t bitSize = size * sizeof(float);
    std::vector<float> output(size);

    float *deviceData = nullptr;
    cudaMalloc(&deviceData, bitSize);

    cudaMemcpy(deviceData, input.data(), bitSize, cudaMemcpyHostToDevice);

    constexpr int numThreads = 256;
    int numBlocks = (static_cast<int>(size) + numThreads - 1) / numThreads;
    geluFunc<<<numBlocks, numThreads>>>(deviceData, static_cast<int>(size));

    cudaMemcpy(output.data(), deviceData, bitSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceData);

    return output;
}