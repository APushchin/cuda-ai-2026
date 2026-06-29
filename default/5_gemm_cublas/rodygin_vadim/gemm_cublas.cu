#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstdio>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix dimensions do not match size n");
    }
    if (n == 0) {
        return std::vector<float>();
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }

    float *d_a, *d_b, *d_c;
    cudaError_t cuda_err;
    
    cuda_err = cudaMalloc(&d_a, n * n * sizeof(float));
    if (cuda_err != cudaSuccess) {
        cublasDestroy(handle);
        throw std::runtime_error("Failed to allocate device memory for matrix A");
    }

    cuda_err = cudaMalloc(&d_b, n * n * sizeof(float));
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cublasDestroy(handle);
        throw std::runtime_error("Failed to allocate device memory for matrix B");
    }

    cuda_err = cudaMalloc(&d_c, n * n * sizeof(float));
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cublasDestroy(handle);
        throw std::runtime_error("Failed to allocate device memory for result matrix C");
    }

    cuda_err = cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
        throw std::runtime_error("Failed to copy matrix A to device");
    }

    cuda_err = cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
        throw std::runtime_error("Failed to copy matrix B to device");
    }

    cuda_err = cudaMemset(d_c, 0, n * n * sizeof(float));
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
        throw std::runtime_error("Failed to initialize result matrix C");
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    status = cublasSgemm(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     n, n, n,
                     &alpha,
                     d_b, n,
                     d_a, n,
                     &beta,
                     d_c, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
        throw std::runtime_error("cuBLAS sgemm computation failed");
    }

    std::vector<float> c(n * n);
    cuda_err = cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
        throw std::runtime_error("Failed to copy result matrix from device");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return c;
}
