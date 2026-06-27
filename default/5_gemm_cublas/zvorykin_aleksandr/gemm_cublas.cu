#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    // Place your implementation here
    size_t size = n * n;
    size_t bytes = size * sizeof(float);
    std::vector<float> c(size);

    float *deviceA = nullptr;
    float *deviceB = nullptr;
    float *deviceC = nullptr;

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, deviceB, n, deviceA, n, &beta, deviceC, n);
   
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), deviceC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cublasDestroy(cublasHandle); 

    return c;
}
