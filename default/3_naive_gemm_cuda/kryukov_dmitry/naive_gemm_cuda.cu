#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

template<typename T>
struct vec_ptrs { T* start; T* finish; T* end; };

template<typename T>
static void set_vec_size(std::vector<T>& v, size_t n) {
    reinterpret_cast<vec_ptrs<T>&>(v).finish = v.data() + n;
}

__global__ void gemm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
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
    }

    MemManager(const MemManager&) = delete;
    MemManager& operator=(const MemManager&) = delete;

    inline void resize(size_t bytes) {
        if (!stream) cudaStreamCreate(&stream);

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

private:
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    size_t d_bytes = 0;
    cudaStream_t stream = nullptr;
};

static MemManager mem;

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t nn = n * n;
    size_t bytes = nn * sizeof(float);
    const int block_size = 16;
    const int grid_size = (n + block_size - 1) / block_size;

    mem.resize(bytes);

    dim3 block(block_size, block_size);
    dim3 grid(grid_size, grid_size);

    cudaMemcpyAsync(mem.a(), a.data(), bytes, cudaMemcpyHostToDevice, mem.s());
    cudaMemcpyAsync(mem.b(), b.data(), bytes, cudaMemcpyHostToDevice, mem.s());
    gemm_kernel<<<grid, block, 0, mem.s()>>>(mem.a(), mem.b(), mem.c(), n);

    std::vector<float> c;
    c.reserve(nn);
    cudaHostRegister(c.data(), bytes, cudaHostRegisterDefault);
    cudaMemcpyAsync(c.data(), mem.c(), bytes, cudaMemcpyDeviceToHost, mem.s());
    cudaStreamSynchronize(mem.s());

    cudaHostUnregister(c.data());
    set_vec_size(c, nn);

    return c;
}
