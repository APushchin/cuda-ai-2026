#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thread>


class CUBLASContext {
public:
    cublasHandle_t handle;
    cudaStream_t stream;
    float* gpu_a = nullptr;
    float* gpu_b = nullptr;
    float* gpu_c = nullptr;
    size_t alloc_bytes = 0;

    CUBLASContext() {
        cublasCreate(&handle);
        cudaStreamCreate(&stream);
        cublasSetStream(handle, stream);
    }

    void ensure_capacity(size_t bytes) {
        if (bytes <= alloc_bytes) {
            return;
        }

        if (gpu_a) {
            cudaFree(gpu_a);
            cudaFree(gpu_b);
            cudaFree(gpu_c);
        }

        cudaMalloc(&gpu_a, bytes);
        cudaMalloc(&gpu_b, bytes);
        cudaMalloc(&gpu_c, bytes);

        alloc_bytes = bytes;
    }

    ~CUBLASContext() {
        if (gpu_a) {
            cudaFree(gpu_a);
            cudaFree(gpu_b);
            cudaFree(gpu_c);
        }
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
    }
};

static CUBLASContext ctx;


std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {

    const size_t elems = static_cast<size_t>(n) * n;
    const size_t bytes = elems * sizeof(float);

    ctx.ensure_capacity(bytes);

    cudaMemcpyAsync(ctx.gpu_a, a.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.gpu_b, b.data(), bytes, cudaMemcpyHostToDevice, ctx.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(ctx.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, n, n,
                &alpha,
                ctx.gpu_b, n,
                ctx.gpu_a, n,
                &beta,
                ctx.gpu_c, n);

    std::vector<float> c;
    std::thread alloc_thread([&c, elems]() {
        c.resize(elems);
    });

    alloc_thread.join();
    cudaMemcpyAsync(c.data(), ctx.gpu_c, bytes, cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    return c;
}
