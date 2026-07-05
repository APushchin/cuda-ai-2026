#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) 
{

    const int mtxNumEl = static_cast<int>(a.size());
    const size_t bitMtxNumEl = mtxNumEl * sizeof(float);

    float *deviceMtxA = nullptr;
    cudaMalloc(&deviceMtxA, bitMtxNumEl);
    cudaMemcpy(deviceMtxA, a.data(), bitMtxNumEl, cudaMemcpyHostToDevice);

    float *deviceMtxB = nullptr;
    cudaMalloc(&deviceMtxB, bitMtxNumEl);
    cudaMemcpy(deviceMtxB, b.data(), bitMtxNumEl, cudaMemcpyHostToDevice);

    float *deviceMtxC = nullptr;
    cudaMalloc(&deviceMtxC, bitMtxNumEl);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    const float alpha(1.0f), beta(0.0f);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, &alpha, 
                deviceMtxB, n, 
                deviceMtxA, n, 
                &beta, deviceMtxC, n);

    std::vector<float> output(mtxNumEl);
    cudaMemcpy(output.data(), deviceMtxC, bitMtxNumEl, cudaMemcpyDeviceToHost);

    cudaFree(deviceMtxC);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxA);
    cublasDestroy(cublasHandle); 

    return output;

}