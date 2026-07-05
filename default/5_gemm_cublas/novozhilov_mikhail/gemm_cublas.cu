
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>
#include <vector>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *gpu_a = nullptr;
    float *gpu_b = nullptr;
    float *gpu_c = nullptr;

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cudaMalloc(&gpu_a, size);
    cudaMalloc(&gpu_b, size);
    cudaMalloc(&gpu_c, size);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaMemcpy(gpu_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b.data(), size, cudaMemcpyHostToDevice);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, gpu_b, n, gpu_a, n, &beta, gpu_c, n);
   
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), gpu_c, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    cublasDestroy(cublasHandle); 

    return c;
}