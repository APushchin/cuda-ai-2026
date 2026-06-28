#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {

    cublasHandle_t handle;
    cublasCreate(&handle);

    int elements_num = a.size();
    std::vector<float> result(a.size());

    float* a_gpu = nullptr;
    float* b_gpu = nullptr;
    float* result_gpu = nullptr;

    size_t size_in_bytes = a.size() * sizeof(float);

    cudaMalloc(&a_gpu, size_in_bytes);
    cudaMalloc(&b_gpu, size_in_bytes);
    cudaMalloc(&result_gpu, size_in_bytes);

    cudaMemcpy(a_gpu, a.data(), size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), size_in_bytes, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, b_gpu, n, a_gpu, n, &beta, result_gpu, n);

    cudaMemcpy(result.data(), result_gpu, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);

    cublasDestroy(handle);

    return result;
}
