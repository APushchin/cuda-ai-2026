#include "gelu_cuda.h"
#include <cuda_runtime.h>

__constant__ float c_k1 = 0.7978845608028654f;
__constant__ float c_k2 = 0.044715f;

__global__ void gelu_kernel(const float* input,
                                 float* output,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float arg = c_k1 * (x + c_k2 * x * x * x);
        float tanh_val = tanhf(arg);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}


std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_input, input.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(d_input, d_output, n);

    std::vector<float> result(n);

    cudaMemcpyAsync(result.data(), d_output, n * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}