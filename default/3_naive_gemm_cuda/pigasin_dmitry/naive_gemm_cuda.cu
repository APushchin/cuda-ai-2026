#include "naive_gemm_cuda.h"


__global__ void kernel(const float* a, const float* b, float* c, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = idx / n;
    const size_t j = idx % n;

    if (i >= n || j >= n) {
        return;
    }

    float sum = 0.f;
    for (size_t k = 0; k < n; ++k) {
        sum += a[i * n + k] * b[k * n + j];
    }
    c[idx] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t numElem = a.size();
    std::vector<float> c(numElem);

    float *gpuA, *gpuB, *gpuC;
    const size_t numBytes = numElem * sizeof(float);
    cudaMalloc(&gpuA, numBytes);
    cudaMalloc(&gpuB, numBytes);
    cudaMalloc(&gpuC, numBytes);

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
    int numBlocks = (numElem + blockSize - 1) / blockSize;

    cudaMemcpy(gpuA, a.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, b.data(), numBytes, cudaMemcpyHostToDevice);
    kernel<<<numBlocks, blockSize>>>(gpuA, gpuB, gpuC, n);
    cudaMemcpy(c.data(), gpuC, numBytes, cudaMemcpyDeviceToHost);

    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);

    return c;
}
