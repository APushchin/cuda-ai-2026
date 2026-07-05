#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

template<typename T>
struct vec_ptrs { T* start; T* finish; T* end; };

template<typename T>
static void set_vec_size(std::vector<T>& v, size_t n) {
    reinterpret_cast<vec_ptrs<T>&>(v).finish = v.data() + n;
}

constexpr int BS = 16;

__global__ void gemm_block_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n)
{
    int row = blockIdx.y * BS + threadIdx.y;
    int col = blockIdx.x * BS + threadIdx.x;

    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < n; t += BS) {
        if (row < n && t + threadIdx.x < n) {
            As[threadIdx.y][threadIdx.x] = a[row * n + t + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (t + threadIdx.y < n && col < n) {
            Bs[threadIdx.y][threadIdx.x] = b[(t + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BS; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
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

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t nn = n * n;
    size_t bytes = nn * sizeof(float);

    mem.resize(bytes);

    dim3 threads(BS, BS);
    dim3 grid(n / BS, n / BS);

    cudaMemcpyAsync(mem.a(), a.data(), bytes, cudaMemcpyHostToDevice, mem.s());
    cudaMemcpyAsync(mem.b(), b.data(), bytes, cudaMemcpyHostToDevice, mem.s());
    gemm_block_kernel<<<grid, threads, 0, mem.s()>>>(mem.a(), mem.b(), mem.c(), n);

    std::vector<float> c;
    c.reserve(nn);
    cudaHostRegister(c.data(), bytes, cudaHostRegisterDefault);
    cudaMemcpyAsync(c.data(), mem.c(), bytes, cudaMemcpyDeviceToHost, mem.s());
    cudaStreamSynchronize(mem.s());

    cudaHostUnregister(c.data());
    set_vec_size(c, nn);

    return c;
}
