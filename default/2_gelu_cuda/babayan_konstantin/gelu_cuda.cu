#define _USE_MATH_DEFINES

#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "gelu_cuda.h"

#define BLOCK_SIZE 256

__global__ void GeluCUDAKernel(const float* __restrict__  in, float* __restrict__ out, int n) {
    const float sqrt_pi = -0.071354816f;
    const float sqrt_pi2 = 22.36386f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = x / (1 + __expf(sqrt_pi * x * (sqrt_pi2 + x * x)));
    }
}

class GeluCUDAHandler {
private:
    cudaStream_t stream;
    float *d_in, *d_out;
    std::vector<float> output;
    size_t memSizeLast;
    float inputLast;

public:
    GeluCUDAHandler() : d_in(nullptr), d_out(nullptr), memSizeLast(0), inputLast(0) {
        cudaStreamCreate(&stream);
    }

    std::vector<float>& execute(const std::vector<float>& input) {
        const size_t memSize = input.size() * sizeof(float);
        if (memSize > memSizeLast) {
            if (d_in) {
                cudaFree(d_in);
                cudaFree(d_out);
            }
            cudaMalloc(&d_in, memSize);
            cudaMalloc(&d_out, memSize);
            output = std::vector<float>(input.size());
            memSizeLast = memSize;
        }
        const uint num_blocks = (input.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMemcpyAsync(this->d_in, input.data(), memSize, cudaMemcpyHostToDevice, stream);
        GeluCUDAKernel<<<num_blocks, BLOCK_SIZE,  0, stream>>>(d_in, d_out, input.size());
        cudaMemcpyAsync(output.data(), d_out, memSize, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
        return output;
    }

    ~GeluCUDAHandler() {
        cudaFree(d_in);
        cudaFree(d_out);

        cudaStreamDestroy(stream);
    }
};

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    static GeluCUDAHandler handler;
    return handler.execute(input);
}
