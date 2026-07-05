#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

template<typename T>
struct vec_ptrs { T* start; T* finish; T* end; };

template<typename T>
static void set_vec_size(std::vector<T>& v, size_t n) {
    reinterpret_cast<vec_ptrs<T>&>(v).finish = v.data() + n;
}

constexpr int BLOCK_SIZE = 256;

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int row_size)
{
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float max_val = -1e30f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = input[row * row_size + i];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float sum_exp = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float val = expf(input[row * row_size + i] - row_max);
        output[row * row_size + i] = val;
        sum_exp += val;
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    if (row_sum == 0.0f) row_sum = 1.0f;

    for (int i = tid; i < row_size; i += blockDim.x) {
        output[row * row_size + i] /= row_sum;
    }
}

class MemManager {
public:
    MemManager() {}
    ~MemManager() {
        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        d_bytes = 0;
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }

    MemManager(const MemManager&) = delete;
    MemManager& operator=(const MemManager&) = delete;

    inline void resize(size_t bytes) {
        if (!stream) cudaStreamCreate(&stream);
        if (bytes == d_bytes) return;

        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        d_bytes = 0;

        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        d_bytes = bytes;
    }

    inline float* input() { return d_input; }
    inline float* output() { return d_output; }
    inline cudaStream_t s() { return stream; }

private:
    float *d_input = nullptr;
    float *d_output = nullptr;
    size_t d_bytes = 0;
    cudaStream_t stream = nullptr;
};

static MemManager mem;

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    size_t nn = input.size();
    size_t bytes = nn * sizeof(float);

    mem.resize(bytes);

    cudaMemcpyAsync(mem.input(), input.data(), bytes, cudaMemcpyHostToDevice, mem.s());

    int row_size = nn / row_count;
    size_t shared_bytes = BLOCK_SIZE * sizeof(float);

    softmax_kernel<<<row_count, BLOCK_SIZE, shared_bytes, mem.s()>>>(
        mem.input(), mem.output(), row_size);

    std::vector<float> output;
    output.reserve(nn);
    cudaHostRegister(output.data(), bytes, cudaHostRegisterDefault);
    cudaMemcpyAsync(output.data(), mem.output(), bytes, cudaMemcpyDeviceToHost, mem.s());
    cudaStreamSynchronize(mem.s());

    cudaHostUnregister(output.data());
    set_vec_size(output, nn);

    return output;
}