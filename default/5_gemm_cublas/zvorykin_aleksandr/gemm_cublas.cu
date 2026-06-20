#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    // Place your implementation here
    const int dataSize = n * n * sizeof(float); 
    cublasHandle_t handle;
    cudaStream_t   stream;
    cublasCreate(&handle);
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    float *d_A;
    cudaMalloc(&d_A, dataSize);
    float *d_B;
    cudaMalloc(&d_B, dataSize);
    float *d_C;
    cudaMalloc(&d_C, dataSize);

    float *h_A;
    cudaHostAlloc(&h_A, dataSize, cudaHostAllocDefault);
    float *h_B;
    cudaHostAlloc(&h_B, dataSize, cudaHostAllocDefault);
    float *h_C;
    cudaHostAlloc(&h_C, dataSize, cudaHostAllocDefault);

    std::memcpy(h_A, a.data(), dataSize);
    std::memcpy(h_B, b.data(), dataSize);

    cudaMemcpyAsync(d_A, h_A, dataSize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, dataSize, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 n, n, n,
                 &alpha,
                 d_B, CUDA_R_32F, n,
                 d_A, CUDA_R_32F, n,
                 &beta,
                 d_C, CUDA_R_32F, n,
                 CUBLAS_COMPUTE_32F_FAST_TF32,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpyAsync(h_C, d_C, dataSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> c(static_cast<size_t>(n * n));
    std::memcpy(c.data(), h_C, dataSize);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
  
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    return c;
}
