#include "gemm_cublas.h"

#include <cuda/cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const std::size_t memsize = a.size() * sizeof(float);

    float *in_a = nullptr;
    float *in_b = nullptr;
    float *out = nullptr;

    cudaMalloc((void**)&in_a, memsize);
    cudaMalloc((void**)&in_b, memsize);
    cudaMalloc((void**)&out, memsize);

    cudaMemcpy(in_a, a.data(), memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(in_b, b.data(), memsize, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, in_b, n, in_a, n, &beta, out, n);

    std::vector<float> result(a.size());

    cudaMemcpy(result.data(), out, memsize, cudaMemcpyDeviceToHost); 

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(out);

    return result;
}