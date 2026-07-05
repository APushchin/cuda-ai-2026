#include "gemm_cublas.h"

#include <cublas_v2.h>

#include <stdexcept>

static void checkStatus(cublasStatus_t st) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(cublasGetStatusString(st));
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const int size = a.size() * sizeof(float);

    float* dev_ptr = nullptr;
    cudaMalloc(&dev_ptr, 3 * size);

    float* a_dev = dev_ptr;
    float* b_dev = dev_ptr + a.size();
    float* c_dev = dev_ptr + 2 * a.size();

    cudaMemcpy(a_dev, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(c_dev, 0, size);

    cublasHandle_t handle;
    checkStatus(cublasCreate(&handle));

    const float alpha = 1.f;
    const float beta = 0.f;
    checkStatus(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b_dev, n, a_dev, n, &beta, c_dev, n));

    std::vector<float> c(a.size());
    float* res = c.data();

    cudaMemcpy(res, c_dev, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);

    checkStatus(cublasDestroy(handle));

    return c;
}
