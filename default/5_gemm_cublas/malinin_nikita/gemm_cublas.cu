#include "gemm_cublas.h"
#include <cublas_v2.h>


std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
        
        int size = a.size();
        int bytesSize = size * sizeof(float);

        float* gpuBufferA = nullptr;
        float* gpuBufferB = nullptr;
        float* gpuBufferC = nullptr;

        cudaMalloc(&gpuBufferA, bytesSize);
        cudaMalloc(&gpuBufferB, bytesSize);
        cudaMalloc(&gpuBufferC, bytesSize);

        cudaMemcpy(gpuBufferA, a.data(), bytesSize, cudaMemcpyHostToDevice);
        cudaMemcpy(gpuBufferB, b.data(), bytesSize, cudaMemcpyHostToDevice);

        float one = 1.0f;
        float zero = 0.0f;

        cublasHandle_t cublas;
        cublasCreate(&cublas);

        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, gpuBufferB, n, gpuBufferA, n, &zero, gpuBufferC, n);

        std::vector<float> c(size);

        cudaMemcpy(c.data(), gpuBufferC, bytesSize, cudaMemcpyDeviceToHost);

        cudaFree(gpuBufferA);
        cudaFree(gpuBufferB);
        cudaFree(gpuBufferC);

        cublasDestroy(cublas);

        return c;
}
