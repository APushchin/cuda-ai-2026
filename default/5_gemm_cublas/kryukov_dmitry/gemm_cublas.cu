#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>

template<typename T>
struct vec_ptrs { T* start; T* finish; T* end; };

template<typename T>
static void set_vec_size(std::vector<T>& v, size_t n) {
    reinterpret_cast<vec_ptrs<T>&>(v).finish = v.data() + n;
}

class MemManager {
public:
    MemManager() {}
    ~MemManager() {
        if (d_a) { cudaFree(d_a); d_a = nullptr; }
        if (d_b) { cudaFree(d_b); d_b = nullptr; }
        if (d_c) { cudaFree(d_c); d_c = nullptr; }
        d_bytes = 0;
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        if (handle) { cublasDestroy(handle); handle = nullptr; }
    }

    MemManager(const MemManager&) = delete;
    MemManager& operator=(const MemManager&) = delete;

    inline void resize(size_t bytes) {
        if (!stream) {
            cudaStreamCreate(&stream);
            cublasCreate(&handle);
            cublasSetStream(handle, stream);
        }
        if (bytes == d_bytes) return;

        if (d_a) { cudaFree(d_a); d_a = nullptr; }
        if (d_b) { cudaFree(d_b); d_b = nullptr; }
        if (d_c) { cudaFree(d_c); d_c = nullptr; }
        d_bytes = 0;

        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        d_bytes = bytes;
    }

    inline float* a() { return d_a; }
    inline float* b() { return d_b; }
    inline float* c() { return d_c; }
    inline cudaStream_t s() { return stream; }
    inline cublasHandle_t h() { return handle; }

private:
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    size_t d_bytes = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
};

static MemManager mem;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t nn = n * n;
    size_t bytes = nn * sizeof(float);

    mem.resize(bytes);

    cudaMemcpyAsync(mem.a(), a.data(), bytes, cudaMemcpyHostToDevice, mem.s());
    cudaMemcpyAsync(mem.b(), b.data(), bytes, cudaMemcpyHostToDevice, mem.s());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(mem.h(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, n, n,
                &alpha,
                mem.b(), n,
                mem.a(), n,
                &beta,
                mem.c(), n);

    std::vector<float> c;
    c.reserve(nn);
    cudaHostRegister(c.data(), bytes, cudaHostRegisterDefault);
    cudaMemcpyAsync(c.data(), mem.c(), bytes, cudaMemcpyDeviceToHost, mem.s());
    cudaStreamSynchronize(mem.s());

    cudaHostUnregister(c.data());
    set_vec_size(c, nn);

    return c;
}