#include "gelu_cuda.h"

#include <cmath>


__device__ inline float fastTanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(x * 2.f)));
}

__device__ inline float fastGelu(float x) {
    return 0.5f * x * (1.0f + fastTanh(0.79788456080286541f * (x + 0.044715f * x * x * x)));
}

__global__ void kernel(float* x, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        x[i] = fastGelu(x[i]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

    float* gpuMem;
    const size_t numBytes = n * sizeof(float);
    cudaMalloc(&gpuMem, numBytes);

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaMemcpy(gpuMem, input.data(), numBytes, cudaMemcpyHostToDevice);
    kernel<<<numBlocks, blockSize>>>(gpuMem, n);
    cudaMemcpy(output.data(), gpuMem, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuMem);
    return output;
}
