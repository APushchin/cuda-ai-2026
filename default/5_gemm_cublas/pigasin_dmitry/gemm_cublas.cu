#include "gemm_cublas.h"

#include <cublas_v2.h>

#include <vector>
#include <thread>


class Ctx {
public:
    cublasHandle_t handle;
    float *gpuA, *gpuB, *gpuC;
    size_t allocBytes = 0;

    Ctx() {
        cublasCreate(&handle);
    }

    void prepareMem(const size_t bytes) {
        if (bytes <= allocBytes)
            return;

        if (gpuA)
            cudaFree(gpuA);
        if (gpuB)
            cudaFree(gpuB);
        if (gpuC)
            cudaFree(gpuC);

        cudaMalloc(&gpuA, bytes);
        cudaMalloc(&gpuB, bytes);
        cudaMalloc(&gpuC, bytes);

        allocBytes = bytes;
    }

    ~Ctx() {
        cublasDestroy(handle);
        cudaFree(gpuA);
        cudaFree(gpuB);
        cudaFree(gpuC);
    }
};

static Ctx ctx;

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c;
    const size_t numElem = a.size();
    std::thread t([&](){c.resize(numElem);});

    const size_t numBytes = numElem * sizeof(float);
    ctx.prepareMem(numBytes);

    cudaMemcpy(ctx.gpuA, a.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.gpuB, b.data(), numBytes, cudaMemcpyHostToDevice);

    float alpha = 1.f;
    float beta = 0.f;
    cublasSgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, ctx.gpuB, n, ctx.gpuA, n, &beta, ctx.gpuC, n);

    t.join();
    cudaMemcpy(c.data(), ctx.gpuC, numBytes, cudaMemcpyDeviceToHost);

    return c;
}
